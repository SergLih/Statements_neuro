from django.db import models
from pytils.translit import slugify
from any_imagefield.models import AnyImageField
from sorl.thumbnail import ImageField
import os

class Group(models.Model):
    name = models.CharField(max_length=128)
    def __str__(self): # For Python 2, use __unicode__ too
        return self.name

class Course(models.Model):
    name = models.CharField(max_length=128)
    def __str__(self): # For Python 2, use __unicode__ too
        return self.name

class Lecturer(models.Model):
    last_name = models.CharField(max_length=128)
    first_name = models.CharField(max_length=128)
    middle_name = models.CharField(max_length=128)
    courses = models.ManyToManyField(Course)
    def __str__(self): # For Python 2, use __unicode__ too
        return "{} {} {}".format(self.last_name, self.first_name, self.middle_name)

class Student(models.Model):
    last_name = models.CharField(max_length=128)
    first_name = models.CharField(max_length=128)
    middle_name = models.CharField(max_length=128, blank=True)
    short_name       = models.SlugField(unique=True, default='')
    #url = models.URLField()
    group = models.ForeignKey(Group, related_name='students', on_delete=models.SET_NULL, null=True)

    class Meta:
        ordering = ['last_name', 'first_name', 'middle_name']

    def save(self, *args, **kwargs):
        self.short_name = slugify(self.last_name)
        print(self.short_name)
        super(Student, self).save(*args, **kwargs)

    def __str__(self): # For Python 2, use __unicode__ too
        return "{} {} {} ({})".format(self.last_name, self.first_name, self.middle_name, self.group.name)

class Lecture(models.Model):
    def get_upload_path(instance, filename): # аргументы обязаны быть такими
        return os.path.join("lectures", "group"+str(instance.group.id), "lecture_" + instance.date_time.strftime("%Y%m%d_%H%M") + "_" + filename)
    #name = models.CharField(max_length=128, unique=True)
    date_time = models.DateTimeField(auto_now=True)
    subject   = models.ForeignKey(Course, on_delete=models.SET_NULL, null=True)
    topic     = models.CharField(max_length=128)
    photo_aud = models.ImageField(upload_to=get_upload_path)
    photo_faces = models.ImageField(blank=True)
    group = models.ForeignKey(Group, on_delete=models.SET_NULL, null=True)
    lecturer = models.ForeignKey(Lecturer, on_delete=models.SET_NULL, null=True)
    students = models.ManyToManyField(Student, blank=True)
    def __str__(self): # For Python 2, use __unicode__ too
        return "{} {}".format(self.subject, self.topic)



class StudentPhoto(models.Model):
	def get_upload_path(instance, filename): # аргументы обязаны быть такими
		return os.path.join("students", "group"+str(instance.student.group.id), instance.student.short_name, filename)

	student =  models.ForeignKey(Student, related_name='photos', on_delete=models.SET_NULL, null=True)
	photo = ImageField(upload_to=get_upload_path, blank=True, null=True)
