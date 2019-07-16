import socket
import json
from state_check.models import *
from state_check.forms import *
from django.conf import settings
from django.db import IntegrityError

from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.forms import modelformset_factory

SIGNATURE = b'lihrazum9876543210!@#$%^&*()'
HOST = 'localhost'
PORT = 6788
ADDR = (HOST,PORT)
BUFSIZE = 4096
LS = len(SIGNATURE)



def index(request):
    return render(request, 'state_check/index.html')#, context_dict)

def netcat(hostname, port, content):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((hostname, port))
    s.sendall(content.encode(encoding='utf-8'))
    s.shutdown(socket.SHUT_WR)
    while 1:
        data = s.recv(1024)
        if data != "":
            s.close()
            return data

def show_student(request, short_name):
    try:
        student = Student.objects.get(short_name=short_name)
        # context_dict['student'] = student

    except Student.DoesNotExist:
        raise Http404("Студент не найден")

    return render(request, 'state_check/show_student.html', {'student': student})


def add_student(request):
    #ImageFormset = modelformset_factory(StudentPhoto, fields=('photo',), extra=10)
    form = StudentForm()
    # A HTTP POST?
    if request.method == 'POST':
        form = StudentForm(request.POST, request.FILES)
        #formset = ImageFormset(request.POST or None, request.FILES or None)
        # Have we been provided with a valid form?
        if form.is_valid(): #and formset.is_valid():
            # Save the new category to the database.
            student = form.save(commit=True)
            print(student)
            for file in request.FILES.getlist('images'):
                try:
                    photo = StudentPhoto(student=student, photo=file)
                    photo.save()
                except Exception as e:
                    break
            print('everything saved!')
            # try:
            #     response = netcat(settings.GODAEMON_ADDRESS, settings.GODAEMON_PORT,
            #                     'CLASSIFY\n{}\n'.format(stems)).decode('utf-8').splitlines()
            #     status, cat = response[:2]
            #
            # except ConnectionRefusedError:
            #     status, cat = 'ERROR', 'Нет соединения с сервером'

            # Now that the category is saved
            # We could give a confirmation message
            # But since the most recent category added is on the index page
            # Then we can direct the user back to the index page.
            return redirect('show_student', short_name=student.short_name)
            # return render(request, 'state_check/show_student.html', {'student': student})
        else:
            # The supplied form contained errors -
            # just print them to the terminal.
            print(form.errors)
    else:
        # Will handle the bad form, new form, or no form supplied cases.
        # Render the form with error messages (if any).
        form = StudentForm()
        #formset = ImageFormset(queryset=StudentPhoto.objects.none())
        #context = {
        #    'form': form,
            #'formset': formset,
        #}
        return render(request, 'state_check/add_student.html', {'form': form})


def add_lecturer(request):
    form = LecturerForm()
    # A HTTP POST?
    if request.method == 'POST':
        form = LecturerForm(request.POST, request.FILES)
        # Have we been provided with a valid form?
        if form.is_valid():
            # Save the new category to the database.
            form.save(commit=True)
            # Now that the category is saved
            # We could give a confirmation message
            # But since the most recent category added is on the index page
            # Then we can direct the user back to the index page.
            return index(request)
        else:
            # The supplied form contained errors -
            # just print them to the terminal.
            print(form.errors)

    # Will handle the bad form, new form, or no form supplied cases.
    # Render the form with error messages (if any).
    return render(request, 'state_check/add_lecturer.html', {'form': form})


def add_lecture(request):
    def get_faces_path(photo_aud_path):
        path_parts = list(os.path.split(photo_aud_path))
        path_parts[-1] = 'faces_' + path_parts[-1]
        return os.path.join(*path_parts)

    form = LectureForm()
    if request.method == 'POST':
        form = LectureForm(request.POST, request.FILES)
        if form.is_valid():
            lecture = form.save(commit=True)
            print(lecture)
            print('everything saved!')

            b = bytes(SIGNATURE + b'c' + lecture.group.id.to_bytes(4, byteorder='big')
                        + lecture.photo_aud.url[7:].encode('ascii'))      #    o_O
            print('sent',len(b),'bytes')

            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                client.connect(ADDR)
                client.send(b)
                print('sent')
                data = client.recv(BUFSIZE)
                persons = json.loads(data[LS:].decode('ascii'))
                print('recieved: ', persons)
                for short_name in persons:
                    student =  Student.objects.get(short_name=short_name)
                    lecture.students.add(student)
                lecture.photo_faces = get_faces_path(lecture.photo_aud.url[7:])
                lecture.save()
                client.close()
            except ConnectionRefusedError:
                print('[ERROR]\tServer has not responded')
            except JSONDecodeError:
                print('[ERROR]\tSomething went wrong on the server')



            return redirect('show_lecture', id=lecture.id)
        else:
            print(form.errors)
    else:
        form = LectureForm()
        return render(request, 'state_check/add_lecture.html', {'form': form})

def edit_attendance(request, id):
    try:
        lecture = Lecture.objects.get(id=id)
        data = dict(request.POST)
        lecture.students.clear()
        for short_name in data.keys():
            if short_name not in ['csrfmiddlewaretoken', 'submit']:
                student =  Student.objects.get(short_name=short_name)
                lecture.students.add(student)
        lecture.save()

        return redirect('show_lecture', id=lecture.id)
    except Lecture.DoesNotExist:
        raise Http404("Лекция не найдена")

def show_lecture(request, id):
    try:
        lecture = Lecture.objects.get(id=id)


    except Lecture.DoesNotExist:
        raise Http404("Лекция не найдена")

    return render(request, 'state_check/show_lecture.html', {'lecture': lecture})
