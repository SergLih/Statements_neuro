# Generated by Django 2.2.3 on 2019-07-14 12:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('state_check', '0017_auto_20190714_1439'),
    ]

    operations = [
        migrations.AddField(
            model_name='lecture',
            name='photo_faces',
            field=models.ImageField(blank=True, upload_to=''),
        ),
    ]