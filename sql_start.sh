#!/bin/bash

./cloud_sql_proxy -instances=$PROJECT_NAME:$PROJECT_REGION:$POSTGRES_INSTANCE=tcp:5432

