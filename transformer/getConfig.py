# coding=utf-8
import configparser

def get_config(file_path='config.ini'):
    parser=configparser.ConfigParser()
    parser.read(file_path)

    _strings=[(key,str(value)) for key,value in parser.items('strings')]
    _ints=[(key,int(value)) for key,value in parser.items('ints')]
    _floats=[(key,float(value)) for key,value in parser.items('floats')]

    return dict(_strings+_ints+_floats)

