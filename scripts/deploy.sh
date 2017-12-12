#!/bin/bash

stack build
stack exec site clean
stack exec site build
rm -rf $PUBLIC_BLOG/*
cp -a _site/* $PUBLIC_BLOG/
