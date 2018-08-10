# =====================================================================
# Title: Useful Functions
# Author: Rio Branham 
# =====================================================================

# %% Imports

import re
import os
import sys
import time
import shelve
import smtplib
import email.utils

import pandas as pd
import pygsheets as pg

from mimetypes import guess_type
from email.encoders import encode_base64
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# %% Function Definitions


def search_text_files(file_paths, string, case_sensitive=False,
                      show_lines=False):
    """
    Search for text within text files
    and return list of files with text present.

    If show_lines=True (default is False) then the line(s) where
    `string` is found will be output to the console
    """
    # Initialize return list
    files_with_text = []

    # Loop through files and search for string
    for path in file_paths:
        # Read in file
        f = open(path, 'r')
        try:
            file = f.read() if case_sensitive else f.read().lower()

        except UnicodeDecodeError:
            print('{} could not be decoded'.format(path))

        finally:
            f.close()

        if re.search(string if case_sensitive else string.lower(), file):
            files_with_text.append(path)

        if show_lines:
            f = open(path, 'r')
            try:
                file_list = f.readlines()

            except UnicodeDecodeError:
                print('{} could not be decoded'.format(path))

            finally:
                f.close()

            if not case_sensitive:
                file_list = [line.lower() for line in file_list]

            found_lines = [line for line in file_list if
                           re.search(string if case_sensitive else
                                     string.lower(), line)]

            if len(found_lines):
                if input('Show lines for {}, {} lines found? (y/n): '.
                         format(path, len(found_lines))).lower() == 'y':
                    for line in found_lines:
                        print('\n' + line, end='')

                        if len(file_list) > 1:
                            if input('Next Line? (y/n): ').lower() != 'y':
                                break

    return files_with_text


def search_files(file_pattern='.*\\.py$', string=None, top='.', recursive=True,
                 case_sensitive=False, abspath=True, open_files=False,
                 show_lines=False):
    """
    Search for and return files by pattern in file name.
    The default is to search recursively from the top
    folder, is case insensitive and returns absolutes pathnames.

    Optionally you can supply a search string that for which the files
    with the supplied pattern will be searched in their text and
    paths returned.

    Aditionally if open_files=True (default is False) then the found
    files will be opened using os.startfile.

    Finally, if show_lines=True (default is False) the line(s) in
    which `string` is found will be output to the console. Ignored if
    `string` is None.
    """
    if not case_sensitive:
        file_pattern = file_pattern.lower()

    # Initialize Results List
    found_files = []

    # Search directory
    if recursive:
        for path, dirs, files in os.walk(top=top):
            for file in files:
                if re.search(file_pattern,
                             file if case_sensitive else file.lower()):
                    if abspath:
                        found = os.path.abspath(os.path.join(path, file))

                    else:
                        found = os.path.join(path, file)

                    found_files.append(found)

    else:
        for item in os.listdir(top):
            if re.search(file_pattern,
                         item if case_sensitive else item.lower()):
                if abspath:
                    found = os.path.abspath(os.path.join(top, item))

                else:
                    found = item

                found_files.append(found)

    # Find Files that contain string
    if string is not None:
        found_files = search_text_files(found_files, string, case_sensitive,
                                        show_lines)

    if open_files:
        for file in found_files:
            os.startfile(file)

    return found_files


def get_mimetype(filename):
    """
    Author: JD Lauret

    Returns the MIME type of the given file.

    :param filename: A valid path to a file
    :type filename: str

    :returns: The file's MIME type
    :rtype: tuple
    """
    content_type, encoding = guess_type(filename)
    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
    return content_type.split('/', 1)


def mimify_file(filename):
    """
    Author: JD Lauret

    Returns an appropriate MIME object for the given file.

    :param filename: A valid path to a file
    :type filename: str

    :returns: A MIME object for the given file
    :rtype: instance of MIMEBase
    """
    filename = os.path.abspath(os.path.expanduser(filename))
    base_file_name = os.path.basename(filename)

    msg = MIMEBase(*get_mimetype(filename))
    msg.set_payload(open(filename, 'rb').read())
    msg.add_header('Content-Disposition',
                   'attachment',
                   filename=base_file_name)

    encode_base64(msg)

    return msg


def mail(to, subject, msg, files=None):
    """
    Easily send an html email using a list of `to` addresses a
    subject and a message and optionally a list of files to attach
    """
    name = 'Your Name'
    sender = 'Your Email'

    m = MIMEMultipart()
    m.attach(MIMEText(msg, 'html'))
    m['From'] = email.utils.formataddr((name, sender))
    m['Subject'] = subject
    m['Bcc'] = m['From']

    for i in to:
        m['To'] = email.utils.formataddr(('', i))

    # Use shelve file to store Gmail credentials
    with shelve.open(creds_file) as creds:
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(creds['gid'], creds['gpd'])

    if files is not None:
        [m.attach(mimify_file(file)) for file in files]
    s.sendmail(sender, to, m.as_string())
    s.quit()


def new_project(project_name, parent_dir='.'):
    """
    Create a new directory with basic project folders.
    """
    os.mkdir(os.path.join(parent_dir, project_name))

    project = os.path.join(parent_dir, project_name)

    for i in ['code', 'bin', 'data', 'outputs']:
        os.mkdir(os.path.join(project, i))


def pyg_upload(ss, sheet_name, data):
    """
    Quickly upload a dataset to a googlesheet worksheet by worksheet
    name
    """
    ws = ss.worksheet('title', sheet_name)
    ws.clear()
    ws.set_dataframe(data, start='A1', fit=True)
