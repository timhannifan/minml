import click
import psycopg2 as pg
import csv
import pandas as pd
import json


RESULTS_FIELDS = 'train_start,train_end,test_start,test_end,model_name,params,metric,threshold,metric_value'


class DBEngine:
    def __init__(self, exp_config, db_config):
        self.dbname = db_config['db']
        self.dbhost = db_config['host']
        self.dbport = db_config['port']
        self.dbusername = db_config['user']
        self.dbpasswd = db_config['pass']
        self.conn = None
        self.experiment_config = exp_config
        self.config = db_config

        self.config_path = self.experiment_config['config_path']
        self.clean_sql = self.config_path + 'config_db_clean.sql'
        self.semantic_sql = self.config_path + 'config_db_semantic.sql'
        self.insert_sql = self.config_path + 'config_db_insert.sql'
        self.drop_and_create_sql = self.config_path + 'config_db_create.sql'
        self.index_sql = self.config_path + 'config_db_index.sql'
        self.data_path = self.experiment_config['input_path']

    def get_split(self, start, end):
        cur = self.get_db_cursor()

        cmd = """
        select *
        from semantic.events
        where date::timestamp between '%s' and '%s';
        """ %(start, end)

        cur.execute(cmd)
        results = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]
        self.close_connection()

        return (column_names, results)


    # Create any tables needed by this Client. Drop table if exists first.
    def create_tables(self):
        click.echo(f"Creating tables")
        dc_commands = get_sql_contents(self.drop_and_create_sql)
        self._execute_sql(dc_commands)


    # Add at least two indexes to the tables to improve analytic queries.
    def add_indices(self):
        click.echo(f"Adding Indexes")
        index_commands = get_sql_contents(self.index_sql)
        self._execute_sql(index_commands)


    # This function will bulk load the data using copy
    def bulk_load_file(self):
        click.echo(f"Bulk load file")
        cur = self.get_db_cursor()

        copy_sql = """
           COPY raw FROM stdin WITH CSV HEADER
           DELIMITER as ','
           """

        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            cur.copy_expert(sql=copy_sql, file=f)
            insert_commands = get_sql_contents(self.insert_sql)
            self._execute_sql(insert_commands)

        self.close_connection()


    def write_result(self, row):
        """
        Writes a row from model evaluation to the results table
        Inputs:
            - row (list): list of results row vals. see db_create.sql schema
        Returns:
            Nothing
        """
        write_row = []
        for el in row:
            if type(el) == dict:
                write_row.append(json.dumps(el))
            else:
                write_row.append(str(el))

        cur = self.get_db_cursor()
        cmd = 'INSERT INTO results'+ \
              ' (%s) VALUES %s ON CONFLICT DO NOTHING;' % (RESULTS_FIELDS,
                                                           tuple(write_row))
        self._execute_sql([cmd])

    def clean_raw(self):
        commands = get_sql_contents(self.clean_sql)
        self._execute_sql(commands)


    def generate_events_entities(self):
        commands = get_sql_contents(self.semantic_sql)
        self._execute_sql(commands)


    def _execute_sql(self, commands):
        # pass a list of sql commands
        cur = self.get_db_cursor()
        for command in commands:
            if command != "":
                try:
                    cur.execute(command)
                    self.conn.commit()
                except (Exception, pg.DatabaseError) as error:
                    print(error)

        self.close_connection()


    # open a connection to a psql database, using the self.dbXX parameters
    def open_connection(self):
        self.conn = pg.connect(host=self.dbhost, database=self.dbname,
                               user=self.dbusername, port=self.dbport)

    # check whether connection is open
    def is_open(self):
        return (self.conn is not None)

    # Close any active connection to the database
    def close_connection(self):
        if self.is_open():
            self.conn.close()
            self.conn = None

    def get_db_cursor(self):
        '''
        Opens db connection if necessary and returns db cursor
        Returns: cursor
        '''
        if not self.is_open():
           self.open_connection()
        return self.conn.cursor()

    # Runs table creation and data loading
    def run(self):
        self.create_tables()
        self.bulk_load_file()
        self.clean_raw()
        self.generate_events_entities()
        self.add_indices()


def get_sql_contents(fname):
    file = open(fname, 'r')
    sql = file.read()
    file.close()

    return sql.split(';')
