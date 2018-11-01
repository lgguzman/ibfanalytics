# ibfAnalytics

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