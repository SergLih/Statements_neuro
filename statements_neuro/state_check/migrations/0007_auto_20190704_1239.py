# -*- coding: utf-8 -*-
# Generated by Django 1.11.14 on 2019-07-04 12:39
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('state_check', '0006_auto_20190704_1148'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lecture',
            name='subject',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='state_check.Course'),
        ),
    ]