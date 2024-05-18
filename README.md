# Walmart-Retail-Analysis

This repository contains the code and methodology for the Walmart Retail Analysis project, which aims to derive actionable insights from Walmart's retail datasets to optimize marketing strategies and improve profit margins using big data technologies and machine learning algorithms.

## Datasets

The datasets used in this project are available under `/data` or can be accessed online at the following URLs:

1. **Walmart1923 Dataset**: [Data.world - Walmart1923](https://data.world/ahmedmnif150/walmart-retail-dataset)
2. **Walmart1215 Dataset**: [Data.world - Walmart1215](https://data.world/garyhoove470/walmart-retail-dataset)

## Data Ingestion

To ingest the data into HDFS, use the following command:
```bash
hdfs dfs -put <CSVFileName>.csv
```

## Data Cleaning
The following commands are used to compile and execute the Java MapReduce jobs for cleaning the data:
```bash
hdfs dfs -put Clean.java CleanMapper.java
hdfs dfs -put CleanReducer.java
javac -classpath `yarn classpath` -d . CleanMapper.java
javac -classpath `yarn classpath` -d . CleanReducer.java
javac -classpath `yarn classpath`:. -d . Clean.java
jar -cvf Clean.jar *.class
hadoop jar Clean.jar Clean <CSVFileName>.csv <DirToStore>
hdfs dfs -ls <DirToStore>
hdfs dfs -cat <DirToStore>/part-r-00000
```

## Data Profiling
The data profiling stage utilizes the following MapReduce Java commands:
```bash
hdfs dfs -put UniqueRecs.java
hdfs dfs -put UniqueRecsMapper.java
hdfs dfs -put UniqueRecsReducer.java
javac -classpath `yarn classpath` -d . UniqueRecsReducer.java
javac -classpath `yarn classpath` -d . UniqueRecsMapper.java
javac -classpath `yarn classpath`:. -d . UniqueRecs.java
jar -cvf Profile.jar *.class
hadoop jar Profile.jar Profile <CSVFileName>.csv <DirToStore>
hdfs dfs -ls <DirToStore>
hdfs dfs -cat <DirToStore>/part-r-00000
```

## Analysis
After data ingestion, cleaning, and profiling, rename datasets as Walmart1215.csv and Walmart1923.csv. Upload the cleaned dataset to DataProc and proceed with the analysis using the following command:
```bash
python train.py
```

