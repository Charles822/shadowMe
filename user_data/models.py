from django.db import models

class UserData(models.Model):
	telegram_id = models.BigIntegerField(unique=True, null=True)
	first_name = models.CharField(max_length=100, null=True)
	last_name = models.CharField(max_length=100, null=True)
	username = models.CharField(max_length=100)

	def __str__(self):
		return f"{self.first_name} {self.last_name}"

