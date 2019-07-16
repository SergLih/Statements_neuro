from django import forms
from django.contrib.admin.widgets import FilteredSelectMultiple
from state_check.models import Student, Lecture, Group, Course, Lecturer

class LectureForm(forms.ModelForm):

    subject   = forms.ModelChoiceField(queryset=Course.objects.all(), to_field_name="name", help_text="Предмет")
    topic     = forms.CharField(max_length=128, help_text="Тема лекции")
    lecturer  = forms.ModelChoiceField(queryset=Lecturer.objects.all(), help_text="Преподаватель")
    photo_aud = forms.ImageField(help_text="Фото аудитории")
    group = forms.ModelChoiceField(queryset=Group.objects.all(), to_field_name="name", help_text="Учебная группа")

    #slug = forms.CharField(widget=forms.HiddenInput(), required=False)
    # An inline class to provide additional information on the form.
    class Meta:
        # Provide an association between the ModelForm and a model
        model = Lecture
        fields = ('subject', 'topic', 'lecturer', 'group', 'photo_aud',)


class StudentForm(forms.ModelForm):

    last_name = forms.CharField(max_length=128, help_text="Фамилия")
    first_name = forms.CharField(max_length=128, help_text="Имя")
    middle_name = forms.CharField(max_length=128, help_text="Отчество")
    group = forms.ModelChoiceField(queryset=Group.objects.all(), to_field_name="name", help_text="Учебная группа")

    #image_st = forms.ImageField(help_text="Фотография")

    class Meta:
        # Provide an association between the ModelForm and a model
        model = Student
        # What fields do we want to include in our form?
        # This way we don't need every field in the model present.
        # Some fields may allow NULL values, so we may not want to include them.
        # Here, we are hiding the foreign key.
        # we can either exclude the category field from the form,
        exclude = ('short_name', )
        # or specify the fields to include (i.e. not include the category field)
        # fields = ('last_name', 'first_name', 'middle_name', 'group', 'image_st',)


class LecturerForm(forms.ModelForm):

    last_name = forms.CharField(max_length=128, help_text="Фамилия")
    first_name = forms.CharField(max_length=128, help_text="Имя")
    middle_name = forms.CharField(max_length=128, help_text="Отчество")
    courses = forms.ModelMultipleChoiceField(queryset=Course.objects.all(),
                                       widget=FilteredSelectMultiple('Courses', False),
                                       required=False)

    class Meta:
        # Provide an association between the ModelForm and a model
        model = Lecturer
        # What fields do we want to include in our form?
        # This way we don't need every field in the model present.
        # Some fields may allow NULL values, so we may not want to include them.
        # Here, we are hiding the foreign key.
        # we can either exclude the category field from the form,
        #exclude = ('category',)
        # or specify the fields to include (i.e. not include the category field)
        fields = ('last_name', 'first_name', 'middle_name', 'courses')
