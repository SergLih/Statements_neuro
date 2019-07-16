from django.conf.urls import url
from state_check import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^student/add/$', views.add_student, name='add_student'),
    url(r'^student/(?P<short_name>[\w\-]+)/$', views.show_student, name='show_student'),
    url(r'^lecturer/add/$', views.add_lecturer, name='add_lecturer'),
    url(r'^lecture/add/$', views.add_lecture, name='add_lecture'),
    url(r'^lecture/(?P<id>\d+)/$', views.show_lecture, name='show_lecture'),
    url(r'^lecture/(?P<id>\d+)/edit_attendance/$', views.edit_attendance, name='edit_attendance'),
]
