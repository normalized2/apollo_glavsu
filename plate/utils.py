# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import scipy.stats as stats
from StringIO import StringIO
from matplotlib import pyplot as plt
from os.path import join, exists, basename
from PIL import Image as pil_image
import re
import requests
import datetime


def download(url, fn_out, timeout_headers=10, maxsize=20000000):
    res = {}
    res['downStatus'] = 'begin'
    res['downHTTPStatus'] = -1
    res['downHTTPContentLength'] = -1
    timeout = timeout_headers
    try:
        timeout_str = 'head'
        headers = {}
        r = requests.head(
            url,
            timeout=timeout,
            headers=headers)

        if r.status_code in [301, 302, 307]:
            timeout_str = 'head2'
            headers = {'referer': url}
            url = r.headers['Location']

            r = requests.head(
                url,
                timeout=timeout,
                headers=headers)

        if r.status_code in [301, 302, 307]:
            timeout_str = 'head3'
            headers = {'referer': url}
            url = r.headers['Location']

            r = requests.head(
                url,
                timeout=timeout,
                headers=headers)

        res['downHTTPStatus'] = r.status_code

        http_keys = ['Content-Length', 'Content-Type', 'Last-Modified']
        db_keys = ['downHTTPContentLength', 'downHTTPContentType',
                   'downHTTPLastModified']
        for k, colname in zip(http_keys, db_keys):
            if k in r.headers:
                res[colname] = r.headers[k]

        if r.status_code == 200:
            size = res['downHTTPContentLength']

            timeout = 3
            # 5 Mb
            if size > 5000000:
                timeout = 5
            # 10 Mb
            if size > 1000000:
                timeout = 10

            if int(size) > maxsize:
                res['downStatus'] = 'skipped'
            else:
                timeout_str = 'get'

                r = requests.get(
                    url,
                    timeout=timeout,
                    headers=headers,
                    stream=True)

                res['downHTTPStatus'] = r.status_code

                if r.status_code == 200:
                    with open(fn_out, 'wb') as f:
                        for chunk in r.iter_content(1024):
                            f.write(chunk)
                res['downStatus'] = 'ok'
        else:
            res['downStatus'] = 'error'

    except IOError as e:
        res['downStatus'] = 'ioerror'
        res['downErrorNo'] = e.errno
        res['downErrorStr'] = e.strerror
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        print(e)
    except requests.exceptions.Timeout as e:
        res['downStatus'] = 'timeout'
        res['downErrorStr'] = timeout_str
    except ValueError as e:
        res['downStatus'] = 'valueError'
        res['downErrorStr'] = str(e)
    return res


df_flickr_photos = pd.read_csv('flickr_photos.csv')

def get_image_url(df=df_flickr_photos, title='AS07-3-1511', site='flikr', size='o', verbose=1):
    rows = df[df.title==title]
    if len(rows) == 0:
        if verbose:
            print(title, 'Not found in csv')
            print(rows)
        return None
    row = rows.iloc[0]
    url = None
    if site=='flikr':
        if size=='o':
            url_templ = "https://c1.staticflickr.com/1/{server}/{id}_{originalsecret}_{size}.jpg"
        elif size is None:
            url_templ = "https://www.flickr.com/photos/projectapolloarchive/{id}/in/album-{album_id}/"
        else:
            url_templ = "https://c1.staticflickr.com/1/{server}/{id}_{secret}_{size}.jpg"
        d = dict(row)
        d['size'] = size
        url = url_templ.format(**d)
    return url
