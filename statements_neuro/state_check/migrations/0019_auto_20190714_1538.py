# Generated by Django 2.2.3 on 2019-07-14 12:38

from django.db import migrations, models
import state_check.models


class Migration(migrations.Migration):

    dependencies = [
        ('state_check', '0018_lecture_photo_faces'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lecture',
            name='photo_aud',
            field=models.ImageField(upload_to=state_check.models.Lecture.get_upload_path),
        ),
    ]
