from django.contrib import admin
from state_check.models import Student, Lecture, Group, Course, Lecturer, StudentPhoto

class StudentAdmin(admin.ModelAdmin):
	#exclude = ('short_name', )
    prepopulated_fields = {"short_name": ("last_name",)}

# Register your models here.
admin.site.register(Student, StudentAdmin)
admin.site.register(Lecture)
admin.site.register(Group)
admin.site.register(Course)
admin.site.register(Lecturer)
admin.site.register(StudentPhoto)
