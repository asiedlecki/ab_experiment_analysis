# AB experiment anlysis
tools used for the analysis:
- python 3.8
- jupyter notebook
- pyspark 2.4.7
- pandas
- scipy

![AB experiment description](photos/experiment_description.png?raw=true "AB experiment description")

Information about the data sets:

***test_data.csv columns:***
```
viewer_id: the ID of the viewer
date: the date the viewer saw a commercial for “US Politics This Week”
tv_make: the make (i.e., brand) of TV
tv_size: the size of the TV in inches (approximately measured as the diagonal of the screen)
uhd_capable: whether the TV is (1) or is not (0) capable of displaying Ultra-High-Definition television content
tv_provider: the cable or satellite TV provider
total_time_watched: the total amount of TV watched (in hours) on the day in the ‘date’ column
watched: whether the viewer watched (1) “US Politics This Week” or not (0)
test: viewers are split into test (1) and control (0) groups; test viewers saw the new commercial with their local Mayor while control viewers saw the old commercial with the Mayor of Los Angeles
```
***viewer_data.csv columns:***
```
viewer_id: the ID of the viewer; same ID as in the test_data.csv file
gender: the viewer’s gender
age: the viewer’s age
city: the viewer’s city
```
# Results
See results below and feel free to look at my .ipynb file.

It took me 4 hours to analyze the experiment and up to 30 minutes to present it in readable form.

### 1. Reproduce results mentioned in task description - without any data cleaning.
- I assume that nulls were included by Channel KLMN's data scientist
```python
+----+-------+------------+-------------------+-------------------+-------------------+--------------------+
|test|watched|all_userdays|conversion         |control_group_value|diff_to_base_pp    |diff_to_base_perc   |
+----+-------+------------+-------------------+-------------------+-------------------+--------------------+
|0   |13448  |213699      |0.0629296346730682 |0.0629296346730682 |0.0                |0.0                 |
|1   |9354   |204327      |0.04577955923593064|0.0629296346730682 |-1.7150075437137555|-0.27252780865860043|
+----+-------+------------+-------------------+-------------------+-------------------+--------------------+
```
- we can already see a huge negative difference (-27%)
- probably no significance test required but let's perform it

##### First, let's check if the sample size is big enough to assure the difference.
The minimum detectable difference is 4.26%, so sample is big enough to run significance test. I always check this first to avoid heavy load in some experiments (especially in continuous measures).

Results of Chi2 signifcance test:

![first Chi2 significance test](photos/first_chi2_significance_test.png?raw=true "Chi2 significance test")

###### Answer 1: The result is negative (significant difference).


### 2. Check what is going on here. Are the commercials with local Mayors really driving a lower fraction of people to watch the show?
##### My hypothesis is that uneven distribution of users (by city) drives this difference.

Let's check!

![differences by city](photos/differences_by_city.png?raw=true "Differences by city")

###### Answer 2a: Commercials with local Mayors may help but this is highly dependent on city. It may be that cities with negative results have unpopular Mayors (just assumption).¶

As we can see there is huge misrepresentation for 3 cities and this probably causes the difference:
- Los Angeles (missing in test group)
- Seattle (overrepresentation in test group)
- Philadelphia (overrepresentation in test group)

##### Let's perform test once again without cities mentioned above.
![differences by city](photos/chi2_significance_test_without_unevenly_distributed_cities.png?raw=true "Chi2 significance test without skewed cities.")

###### Answer 2b: As we can see, removal of unevenly distributed cities shows there is no difference. Therefore we can blame uneven distribution for observed differences.

### 3. As we can clearly see there are some issues with distribution. Let's write a code for an algorithm that returns FALSE if that problem happens again in the future and TRUE otherwise.

```python
def check_if_even_distribution(df, features, experiment_variants_number, accepted_threshold):
    diffs_list = []
    
    for feature in features:
        
        window_spec_check = Window.partitionBy(feature).orderBy('test')
        
        # for difference in count between variants
        diff_count = (df
                      .where(f.col(feature).isNotNull())
                      .groupBy(feature, 'test')
                      .agg(f.count('*').alias('count'))
                      .withColumn('first_variant_count', f.first(f.col('count')).over(window_spec_check))
                      .withColumn('diff_to_first', f.abs(f.col('count') - f.col('first_variant_count')) /
                                  f.col('first_variant_count'))
                      .withColumn('unevenly_distributed', f.col('diff_to_first') > accepted_threshold)
                      .where(f.col('unevenly_distributed'))
                      .count()
                     )
        diffs_list.append(diff_count)
        
        # for lack of this variant of the feature (e.g. triggers on Los Angeles not being in control group)
        variants_missed = (df
                           .where(f.col(feature).isNotNull())
                           .groupBy(feature)
                           .agg(f.countDistinct(f.col('test')).alias('variants_cnt'))
                           .where(f.col('variants_cnt') < experiment_variants_number)
                          ).count()
        diffs_list.append(variants_missed)
    return sum(diffs_list) == 0
```

```python
# should return False
check_if_even_distribution(df=test_data_with_viewer_df,
                           features=test_data_with_viewer_df.columns[-3:],
                           experiment_variants_number = 2,
                           accepted_threshold = 0.1
                          )
```
>False

```python
# should return True
test_data_with_viewer_clean_df = (test_data_with_viewer_df
                                  .where(~f.col('city').isin('Philadelphia', 'Los Angeles', 'Seattle', 'Minneapolis'))
                                 )

check_if_even_distribution(df=test_data_with_viewer_clean_df,
                           features=['city'],
                           experiment_variants_number = 2,
                           accepted_threshold = 0.1
                          )
```
>True

###### Answer 3: The above function returns False if samples are unevenly distributed and True if samples are in similar shape.

Can be used for:
- predefined column names
- all number of variants (also multi-variate experiments)
- different levels of sample size difference acceptance

##### Conclusion
###### Setup of every experiment should be checked before making any conclusion. I have checked gender and city, but there might be problem with other features. The check code I wrote was triggered on feature of age so we know that it was distributed unevenly as well.

###### It is worth mentioning that some experiments may give different results depending on group characteristics. It is good idea to perform significance test for each one of those groups and make business decisions related to particular groups. The commercial may work for Detroit users but rather not for Dallas users.

Thanks!