# Generated by Django 5.1.4 on 2025-02-22 08:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user_data', '0002_userdata_telegram_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userdata',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
    ]
