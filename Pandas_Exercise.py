import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import pandas
import pandas as pd

# The imdb.csv dataset contains Highest Rated IMDb "Top 1000" Titles.

# load imdb dataset as pandas dataframe
df_movies = pd.read_csv("/home/ankur/Documents/imdb_1000.csv")

# show first 5 rows of imdb_df
print(df_movies.head())

# The bikes.csv dataset contains information about the number of bicycles that used certain bicycle lanes in Montreal in the year 2012.
# load bikes dataset as pandas dataframe
d_parser = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
df_bikes = pd.read_csv("/home/ankur/Documents/bikes.csv", sep=';', parse_dates=['Date'], date_parser=d_parser)

# show first 3 rows of bikes_df
print(df_bikes.head(3))

# Selecting columns
# When you read a CSV, you get a kind of object called a DataFrame, which is made up of rows and columns. You get columns out of a DataFrame the same way you get elements out of a dictionary.

# list columns of imdb_df
print(df_movies.columns)

# what are the data types of values in columns
print(df_movies.dtypes)

# list first 5 movie titles
print(df_movies.loc[0:5, ["title"]])

# show only movie title and genre
print(df_movies[["title", "genre"]])

# Understanding columns
# On the inside, the type of a column is pd.Series and pandas Series are internally numpy arrays. If you add .values to the end of any Series, you'll get its internal numpy array.

# show the type of duration column
print(df_movies["duration"].dtypes)

# show duration values of movies as numpy arrays
import numpy as np

duration_movies = np.array(df_movies["duration"])
print(duration_movies)

# Applying functions to columns
# Use .apply function to apply any function to each element of a column.

# convert all the movie titles to uppercase
to_uppercase = lambda x: x.upper()
print(df_movies["title"].apply(to_uppercase).head())

# Plotting a column
# Use .plot() function!
print(df_bikes.head())

# plot the bikers travelling to Berri1 over the year
import matplotlib.pyplot as plt

plt.plot(df_bikes["Date"], df_bikes["Berri1"])
plt.xlabel("Date")
plt.ylabel("Bikers travelling to Berri 1")
plt.title("Bikers Data")
plt.show()

# plot all the columns of bikes_df
plt.plot(df_bikes['Date'], df_bikes.loc[0:, 'Rachel / Papineau':'Pont_Jacques_Cartier'],
         label=['Rachel / Papineau', 'Berri1', 'Maisonneuve_2', 'Maisonneuve_1', 'Brébeuf', 'Parc', 'PierDup',
                'CSC (Côte Sainte-Catherine)', 'Pont_Jacques_Cartier'])
plt.xlabel('Date')
plt.ylabel('Bikers travelling to various places')
plt.title("Bikers data")
plt.show()

# Value counts
# Get count of unique values in a particular column/Series.

# what are the unique genre in imdb_df?
genre_arr = pd.unique(df_movies["genre"])
print(genre_arr)

# plotting value counts of unique genres as a bar chart
genre_count = df_movies['genre'].value_counts()

df_temp = pd.DataFrame(genre_count)
n = len(df_temp.index)
left = np.arange(1, 1 * n + 1, 1)
height = list(df_temp['genre'])
tick_label = df_temp.index.values
plt.bar(left, height, tick_label=tick_label,
        width=0.8, color=['red', 'blue'])
plt.show()

# plotting value counts of unique genres as a pie chart
activities = df_temp.index.values

slices = list(df_temp['genre'])

plt.pie(slices, labels=activities,
        startangle=90, shadow=True,
        radius=1.2, autopct='%1.1f%%')
plt.show()

# Index
# DATAFRAME = COLUMNS + INDEX + ND DATA
# SERIES = INDEX + 1-D DATA
# Index or (row labels) is one of the fundamental data structure of pandas. It can be thought of as an immutable array and an ordered set.
#
# Every row is uniquely identified by its index value.

# show index of bikes_df
print(df_bikes.index.values)

# get row for date 2012-01-01
row_data = (df_bikes['Date'] == '2012-01-01')
print(df_bikes[row_data])

# To get row by integer index:
# Use .iloc[] for purely integer-location based indexing for selection by position.

# show 11th row of imdb_df using iloc
print(df_bikes.iloc[10])

# Selecting rows where column has a particular value

# select only those movies where genre is adventure
row_data = (df_movies['genre'] == 'Adventure')
print(df_movies.loc[row_data])

# which genre has highest number of movies with star rating above 8 and duration more than 130 minutes?
filt1 = (df_movies['star_rating'] > 8) & (df_movies['duration'] > 130)
ans = df_movies.loc[filt1, 'genre'].value_counts()
temp_df = pd.DataFrame(ans)
print(temp_df.index[0])

# Adding a new column to DataFrame

# add a weekday column to bikes_df
df_bikes['Weekday'] = df_bikes['Date'].dt.day_name()
print(df_bikes)

# Deleting an existing column from DataFrame

# remove column 'Unnamed: 1' from bikes_df
df_bikes.drop(columns=['Unnamed: 1'], inplace=True)
print(df_bikes)

# Deleting a row in DataFrame

# remove row no. 1 from bikes_df
print(df_bikes.drop([0]))

# Group By
# Any groupby operation involves one of the following operations on the original object. They are −
#
# Splitting the Object
#
# Applying a function
#
# Combining the results
#
# In many situations, we split the data into sets and we apply some functionality on each subset. In the apply functionality, we can perform the following operations −
#
# Aggregation − computing a summary statistic
#
# Transformation − perform some group-specific operation
#
# Filtration − discarding the data with some condition

# group imdb_df by movie genres
genre_grp = df_movies.groupby(['genre'])
print(genre_grp)

# get crime movies group
print(genre_grp.get_group('Crime'))

# get mean of movie durations for each group
duration_mean = genre_grp['duration'].mean()
print(duration_mean)

# change duration of all movies in a particular genre to mean duration of the group
n = len(df_movies.index)
for i in range(n):
    df_movies.loc[i, 'duration'] = duration_mean[df_movies.loc[i, 'genre']]
print(df_movies)

# drop groups/genres that do not have average movie duration greater than 120.
n = len(df_movies.index)
for i in range(n):
    if df_movies.loc[i, 'duration'] <= 120:
        df_movies.drop(i, inplace=True)
print(df_movies)

# group weekday wise bikers count
weekday_grp = df_bikes.groupby(['Weekday'])

# get weekday wise biker count
day_lst = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#sum_lst = []
#for i in range(7):
 #       sum_lst.append(weekday_grp.get_group(day_lst[i]).sum().sum())
#print(sum_lst)

# plot weekday wise biker count for 'Berri1'
Berri1_sum = []
for i in range(7):
        Berri1_sum.append(weekday_grp.get_group(day_lst[i]).sum().loc['Berri1'])
plt.xlabel('Weekday')
plt.ylabel('Bikers count')
plt.plot(day_lst, Berri1_sum)
plt.title('Berri1')
plt.show()
