import google.auth
from googleapiclient import discovery
from google.cloud import storage
import argparse
import yaml
import logging
import json
import apache_beam as beam
from apache_beam import pvalue
import time
from google.auth.transport import requests
from oauth2client.client import GoogleCredentials
from apache_beam import DoFn, GroupByKey, io, ParDo, Pipeline, PTransform, WindowInto, WithKeys
from apache_beam.options.pipeline_options import PipelineOptions
from datetime import datetime
import random
from apache_beam.io import WriteToText
from apache_beam import DoFn, GroupByKey, io, ParDo, Pipeline, PTransform, WindowInto, WithKeys
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.window import FixedWindows
import random

with open("PubSub_BQ.yaml", 'r') as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.BaseLoader)

class ParseMessage(beam.DoFn):
    OUTPUT_ERROR_TAG = 'error'
    
    def process(self, line):
        """
        Extracts fields from json message
        :param line: pubsub message
        :return: have two outputs:
            - main: parsed data
            - error: error message
        """
        try:
            parsed_row = line # parse json message to corresponding bgiquery table schema
            yield parsed_row
        except Exception as error:
            error_row = error # build you error schema here
            yield pvalue.TaggedOutput(self.OUTPUT_ERROR_TAG, error_row)
        

def run():
    """
    Build and run Pipeline
    :param options: pipeline options
    :param input_subscription: input PubSub subscription
    :param output_table: id of an output BigQuery table
    :param output_error_table: id of an output BigQuery table for error messages
    """

    global projectid
    projectid = cfg['projectid']
    global run_date
    run_date = time.strftime('%Y%m%d')
    global jobname
    global timestmp
    timestmp = str(datetime.now())
    global filename
    jobname = cfg['jobname']
    input_subscription=""
    output_     table = ""
    output_error_table =""
    pipeline_args = ['--project', cfg['projectid'],
                     '--job_name', jobname,
                     '--requirements_file', cfg['requirements_file'],
                     '--save_main_session',
                     '--runner', cfg['runner'],
                     '--staging_location', cfg['dataflow_staging'],
                     '--temp_location', cfg['tempbucket'],
                     '--region', cfg['region'],
                     '--worker_zone', cfg['worker_zone'],
                     #'--worker_machine_type', cfg['worker_machine_type'],
                     '--template_location', cfg['template_location'],
                     '--worker_disk_type', cfg['worker_disk_type'],
                     '--disk_size_gb', cfg['disk_size_gb'],
                     '--no_use_public_ips',
                     '--subnetwork', cfg['subnetwork'],
                     '--service_account_email', cfg['service_account_name']
                     ]

    pipeline_options = PipelineOptions(
            pipeline_args, streaming=True, save_main_session=True
        )


    with beam.Pipeline(options=pipeline_options) as pipeline:
        # Read from PubSub
        rows, error_rows = \
            (pipeline | 'Read from PubSub' >> beam.io.ReadFromPubSub(subscription=input_subscription)
             # Adapt messages from PubSub to BQ table
             | 'Parse JSON messages' >> beam.ParDo(ParseMessage()).with_outputs(ParseMessage.OUTPUT_ERROR_TAG,
                                                                                main='rows')
             )
        rows | 'Write to PubSub Topic' >> beam.io.WriteToPubSub(topic=OUTPUT, with_attributes=False)
             

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run()
    
    

