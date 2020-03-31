
""" Core routines needed for the interface to the Postgresql EDW database. """

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv
from geopandas import GeoDataFrame


load_dotenv(find_dotenv())

EDW_RECORDS = "edw.fivehundredcities_data"
EWD_SHAPES = "exposome_pici.shapefile"


def get_edw_db():
    """ Get a database client.

    Returns:
        PostgresQL cursor object
    """
    url = os.getenv('EDW_HOST')
    dbname = os.getenv('EDW_DATABASE')
    user = os.getenv('EDW_USER')
    pwd = os.getenv('EDW_PASSWD')

    try:
        connection = psycopg2.connect(host=url,
                                      database=dbname,
                                      user=user,
                                      password=pwd)

        # NOTE: creating a names cursor is necessasry when using psycopg to
        # avoid fetching the entire result set to the client
        # ref: https://www.buggycoder.com/fetching-millions-of-rows-in-python-psycopg2/
        cursor = connection.cursor(cursor_factory=RealDictCursor,
                                   name='fetch_results')
    #   print(connection.get_dsn_parameters(),"\n")

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    return cursor, connection


def get_fips_data(fips_ids, disease):
    """ retrieves columns for census tracts identified by the
        fips_place_tract and specified disease

    Args:
        fips_ids (list)
        disease (string)

    Returns:
        (pandas DataFrame)
    """

    cursor = get_edw_db()

    # build case sensitive strings for main variables
    disease_prev = "\"" + disease + "_CrudePrev" + "\""
    disease_conf_low = "\"" + disease + "_Crude95CI_low" + "\""
    disease_conf_high = "\"" + disease + "_Crude95CI_high" + "\""

    pipeline = "SELECT stateabbr, placename, fips_tract, fips_place_tract, " \
               + disease_prev + ", " + disease_conf_low + ", " + disease_conf_high \
               + " FROM " + EDW_RECORDS \
               + " WHERE fips_place_tract IN %(fips)s"

    cursor.execute(pipeline, {'fips': tuple(fips_ids)})
    records = pd.DataFrame(cursor.fetchall())
    return records


def get_region_fips(state, city=None):
    """ retrieves all fips ids for a state.  If city is specific
        only fips ids for that city are returned.  Will eventually
        generalize locale structure

    Args:
        city (string)
        state (string)

    Returns:
        (pd.DataFrame) of fips_place_tract, state, and place
    """
    cursor = get_edw_db()

    pipeline = "SELECT stateabbr, placename, fips_place_tract \
                FROM " + EDW_RECORDS + " WHERE stateabbr= %(state)s"
    params = {'state': state}

    if city:
        city = ' '.join(city.split('_'))
        # deal with special cases
        if 'St ' in city:
            city = city.replace('St ', 'St. ')
        elif ('-' in city) & (city != 'Winston-Salem'):
            city = city.replace("-", "\'")
        elif city == 'Ventura':
            city = 'San Buenaventura (Ventura)'
        # append to query if city option called
        pipeline = pipeline + " AND placename=%(city)s"
        params.update({'city': city})

    cursor.execute(pipeline, params)
    records = pd.DataFrame(cursor.fetchall())
    return records


def get_fips_geometry(fips_ids):
    """ return the geometry shapes for each of the fips specified by ids

    Args:
        fips_ids (list)

    Returns:
        (pandas DataFrame)
    """
    cursor, connection = get_edw_db()

    pipeline = "SELECT shape_id, statefip, fipcode, geoid, geometrywkt," \
               + "startdate, enddate" \
               + " FROM " + EWD_SHAPES \
               + " WHERE geoid IN %(fips)s"

    cursor.execute(pipeline, {'fips': tuple(fips_ids)})
    #cursor.execute(pipeline)
    #records = pd.DataFrame(cursor.fetchmany(10))
    records = pd.DataFrame(cursor.fetchall())
    return records



if __name__ == "__main__":
    #data = get_region_fips('MA')\
    data = get_fips_geometry(['48027020402'])
    print(data.geoid)
    print(len(data))
