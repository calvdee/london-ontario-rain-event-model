#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Your command line contains $# arguments, this script requires 3"
    exit -1
fi

cd src/climate-canada-crawler

scrapy crawl monthly_weather_spider \
  -a run_from=$1 \
  -a run_to=$2 \
  -s FEED_URI=$3