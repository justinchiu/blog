#!/bin/bash

stack exec site clean
stack exec site build
rm -rf $PUBLIC_BLOG/*
cp -ar _sites/* $PUBLIC_BLOG/
