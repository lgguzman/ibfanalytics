# ibfAnalytics
This project creates purchase suggestions as follows:
First it creates a cluster of users by purchase behavior and then for each  user set creates a suggestion list using association rules algorithm.
The user cluster is created by 3 kinds of algorithms: K means, bisection means and gaussian Mixture.
All algorithms and processing are executed by distributed processing using Apache Spark.

ParameterList list:

- user_name 
- password 
- domain
- app_key
- special_box
- cluster_year


Run test:
```python
from ibfanalytics import AssociationRules
ar = AssociationRules()
ar.run_test()
```
or 
```python
from ibfanalytics import UserCluster
user = UserCluster()
user.run_test()
```