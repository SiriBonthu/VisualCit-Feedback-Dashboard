from django.db import models


# Create your models here.

class Annotations(models.Model):
    id = models.IntegerField(primary_key=True)
    url = models.CharField(max_length=100)
    ans = models.IntegerField
    pos_ans = models.IntegerField
