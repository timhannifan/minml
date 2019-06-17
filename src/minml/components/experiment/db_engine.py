import click
import psycopg2 as pg
import csv
import numpy as np

class DBEngine:
    def __init__(self, project_path, data_file_path):
        self.dbname = "timhannifan"
        self.dbhost = "127.0.0.1"
        self.dbport = 5432
        self.dbusername = "timhannifan"
        self.dbpasswd = ""
        self.conn = None
        self.project_path = project_path
        self.clean_sql = self.project_path + 'db_clean.sql'
        self.semantic_sql = self.project_path + 'db_semantic.sql'
        self.insert_sql = self.project_path + 'db_insert.sql'
        self.drop_and_create_sql = self.project_path + 'db_create.sql'
        self.index_sql = self.project_path + 'db_index.sql'
        self.data_path = data_file_path

    def fetch_data(self, start, end):
        cur = self.get_db_cursor()

        cmd = """
        select *
        from semantic.events
        where date::timestamp between '%s' and '%s';
        """ %(start, end)

        cur.execute(cmd)
        results = cur.fetchall()

        arr = np.array(results)
        x = arr[:, 1:-2]
        y = arr[:,-1]

        self.close_connection()

        return (x, y)

    # Create any tables needed by this Client. Drop table if exists first.
    def create_tables(self):
        click.echo(f"Creating tables")
        dc_commands = get_sql_contents(self.drop_and_create_sql)
        # print(dc_commands)
        self.execute_sql(dc_commands)


    # Add at least two indexes to the tables to improve analytic queries.
    def add_indices(self):
        click.echo(f"Adding Indexes")
        index_commands = get_sql_contents(self.index_sql)
        self.execute_sql(index_commands)


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
            self.execute_sql(insert_commands)

        self.close_connection()

    def clean_raw(self):
        commands = get_sql_contents(self.clean_sql)
        self.execute_sql(commands)


    def generate_events_entities(self):
        commands = get_sql_contents(self.semantic_sql)
        self.execute_sql(commands)


    def execute_sql(self, commands):
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
        self.conn = pg.connect(host=self.dbhost, database=self.dbname, user=self.dbusername, port=self.dbport)

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
