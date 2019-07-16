# Generated by Django 2.2.3 on 2019-07-12 12:37

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('state_check', '0013_auto_20190712_1224'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lecture',
            name='group',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='state_check.Group'),
        ),
        migrations.AlterField(
            model_name='lecture',
            name='lecturer',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='state_check.Lecturer'),
        ),
        migrations.AlterField(
            model_name='lecture',
            name='subject',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='state_check.Course'),
        ),
        migrations.AlterField(
            model_name='student',
            name='group',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='state_check.Group'),
        ),
        migrations.AlterField(
            model_name='studentphoto',
            name='student',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='photos', to='state_check.Student'),
        ),
    ]
