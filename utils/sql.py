import sqlite3


def connect_and_execute(sqlite_path, cmd, parameters=None):
    conn = sqlite3.connect(sqlite_path)
    with conn as cur:
        if parameters:
            cur.execute(cmd, parameters)
        else:
            cur.execute(cmd)


def get_placeholders(num, form):
    """
    Example:
        >>> get_placeholders(num=3, form="%s")
        '%s, %s, %s'
    """
    return ' '.join([form + "," for _ in range(num)])[:-1]


def create_index(sqlite_path, table_name, col_name):
    # Warning: This cmd is vulnerable to SQL injection via
    # the table_name and col_name variables.
    idx_name = "idx_" + col_name
    values = [idx_name, table_name, col_name]
    cmd = "CREATE INDEX {} ON {} ({})".format(*values)
    connect_and_execute(sqlite_path, cmd)


def update_table(sqlite_path, table_name, key_value_pairs, where_string=None,
                 where_variables=None):
    # Warning: This cmd is vulnerable to SQL injection via
    # the table_name and col_names variables.
    parameters = [v.__name__ if callable(v) or isinstance(v, type) else v
                  for v in key_value_pairs.values()]
    col_names = list(key_value_pairs.keys())
    placeholders = get_placeholders(len(key_value_pairs), "{} = ?")
    if where_string is None:
        where_string = "WHERE id = ?"
        row_id = key_value_pairs['id']
        where_variables = [row_id]
    cmd = " ".join(["UPDATE {}",
                    "SET {}".format(placeholders),
                    where_string])
    cmd = cmd.format(table_name, *col_names)
    parameters += where_variables
    connect_and_execute(sqlite_path, cmd, parameters)


def insert_into_table(sqlite_path, table_name, key_value_pairs):
    # Warning: This cmd is vulnerable to SQL injection via
    # the table_name and col_names variables.
    col_names = key_value_pairs.keys()
    parameters = list(key_value_pairs.values())
    col_name_placeholders = get_placeholders(len(key_value_pairs), "{}")
    col_name_placeholders = "({})".format(col_name_placeholders)
    parameter_placeholders = get_placeholders(len(key_value_pairs), "?")
    parameter_placeholders = "({})".format(parameter_placeholders)
    cmd = " ".join(["INSERT INTO", table_name, col_name_placeholders,
                    "VALUES", parameter_placeholders]).format(*col_names)
    connect_and_execute(sqlite_path, cmd, parameters)


if __name__ == '__main__':
    sqlite_path = "database.sqlite3"
    cmd = """
          CREATE TABLE trials (
              trial_id INTEGER PRIMARY KEY,
              datetime TEXT,
              epochs INTEGER)
          """
    connect_and_execute(sqlite_path, cmd)
