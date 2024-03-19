#=============================================
### Package Imports
#=============================================
import streamlit as st
import duckdb
from io import BytesIO
import io
import pandas as pd
import numpy as np


#=============================================
### Page Layouting and Title
#=============================================
st.set_page_config(layout="wide")  # for a wider layout
st.title("Census Data Filtering App") # Title for the App



conn = duckdb.connect(database=':memory:', read_only=False)
conn.execute("install AZURE")
conn.execute("load azure")
azure_connection = """create or replace secret secret_census (
type azure,
connection_string 'DefaultEndpointsProtocol=https;AccountName=zus1idohdevv2nwssdl;SharedAccessSignature=sv=2021-04-10&st=2024-03-05T18%3A25%3A45Z&se=2024-03-31T17%3A25%3A00Z&sr=c&sp=rl&sig=ANz%2BpBn9q0OOcjMtLkOVGiEkgD%2BnkrDyybGGcD%2B%2BYFg%3D;EndpointSuffix=core.windows.net'
);"""
conn.execute(azure_connection)




def create_partitions_df(partitions_query):
	conn.execute(partitions_query)
	partition_df = conn.execute("SELECT * FROM main.partitions").df()
	return partition_df




def partition_filter():	
	default_option = 0
	vintage_best_set= [a for a in set(partition_df['vintage_best'])]
	vintage_best_set.append("All vintages")
	vintage_best_set = ["Best"if x == 1 else x for x in vintage_best_set]
	vintage_best_set.remove(0) 
	vintage_best_set = set(vintage_best_set)
	selected_vintage_best = st.sidebar.selectbox("Vintage Best ()", vintage_best_set, index=default_option)
	if selected_vintage_best == "All vintages":
	    selected_vintage_best = "0, 1"
	    selected_vintage_best_list = [0,1]
	else:
	    selected_vintage_best
	    selected_vintage_best = "1"
	    selected_vintage_best_list = [1]


	selected_geography_level = st.sidebar.selectbox("Geography Level", set(partition_df[partition_df['vintage_best'].isin(selected_vintage_best_list)]['geography_level']))
	selected_population_year = st.sidebar.multiselect("Population Year", set(partition_df[partition_df['geography_level'] == selected_geography_level]['population_year']))
	selected_population_year = [str(a) for a in selected_population_year]
	selected_population_year_str = ', '.join(selected_population_year)
	return selected_vintage_best, selected_population_year_str, selected_geography_level
	conn.execute(query, [selected_geography_level])

def age_selection(age_query):
    df_age= conn.execute(age_query).df()
    df_age_set = set(df_age['age_group'])
    df_age_set = [str(a) for a in df_age_set]
    df_age_set = sorted(df_age_set)
    selected_age = st.sidebar.multiselect("Age", df_age_set)
    selected_age = [str(a) for a in selected_age]
    selected_age_string = ', '.join(["'" + element + "'" for element in selected_age])
    return selected_age_string


def race_selection(race_query):
    df_race= conn.execute(race_query).df()
    df_race_set = set(df_race['race'])
    df_race_set = [str(a) for a in df_race_set]
    df_race_set = sorted(df_race_set)
    # df_race_set = set(df_race_set)
    selected_race = st.sidebar.multiselect("Race", df_race_set)
    selected_race = [str(a) for a in selected_race]
    selected_race_string = ', '.join(["'" + element + "'" for element in selected_race])
    return selected_race_string


def ethnicity_selection(ethnicity_query):    
    df_ethnicity= conn.execute(ethnicity_query).df()
    df_ethnicity_set = set(df_ethnicity['ethnicity'])
    df_ethnicity_set = [str(a) for a in df_ethnicity_set]
    df_ethnicity_set = sorted(df_ethnicity_set)
    selected_ethnicity = st.sidebar.multiselect("Ethnicity", df_ethnicity_set)
    selected_ethnicity = [str(a) for a in selected_ethnicity]
    selected_ethnicity_string = ', '.join(["'" + element + "'" for element in selected_ethnicity])
    return selected_ethnicity_string


def sex_seletion(sex_query):
    df_sex = conn.execute(sex_query).df()
    df_sex_set = set(df_sex['gender'])
    df_sex_set = [str(a) for a in df_sex_set]
    df_sex_set = sorted(df_sex_set)
    selected_sex = st.sidebar.multiselect("Gender", set(df_sex['gender']))
    selected_sex = [str(a) for a in selected_sex]
    selected_sex_string = ', '.join(["'" + element + "'" for element in selected_sex])
    return selected_sex_string


def create_pop_sheet(dataframe, groupby_cols, newly_created_cols):
    # dataframe.drop(columns = ['source_id', 'vintage_id'], inplace = True)
    new_groupby_cols = [item for item in groupby_cols if item not in ['source_id', 'vintage_id', 'vintage_type']]
    dataframe = dataframe.groupby(new_groupby_cols)[newly_created_cols].sum().reset_index()
    return dataframe


def source_id_sheet(dataframe, groupby_cols, newly_created_cols):
    new_groupby_cols = [item for item in groupby_cols if item not in ['vintage_id', 'vintage_type']]
    new_groupby_cols.remove('source_id')
    dataframe = dataframe.replace(" ", np.nan)
    dataframe[newly_created_cols] = dataframe[newly_created_cols].apply(lambda x: np.where(x.notna(), dataframe['source_id'], x))    
    dataframe= dataframe.groupby(new_groupby_cols, sort=False)[newly_created_cols].apply(lambda x: x.ffill().bfill()).reset_index()
    dataframe= dataframe.drop_duplicates(subset= (new_groupby_cols))
    dataframe_columns = new_groupby_cols + [item for item in newly_created_cols if item not in new_groupby_cols]
    dataframe.sort_values(by = ['geography_name'], inplace = True)
    dataframe.fillna(' ', inplace=True)
    return dataframe[dataframe_columns]


def vintage_id_sheet(dataframe, groupby_cols, newly_created_cols):
    new_groupby_cols = [item for item in groupby_cols if item not in ['source_id', 'vintage_type']]
    new_groupby_cols.remove('vintage_id')
    dataframe = dataframe.replace(" ", np.nan)
    dataframe[newly_created_cols] = dataframe[newly_created_cols].apply(lambda x: np.where(x.notna(), dataframe['vintage_id'], x))
    dataframe= dataframe.groupby(new_groupby_cols, sort=False)[newly_created_cols].apply(lambda x: x.ffill().bfill()).reset_index()
    dataframe= dataframe.drop_duplicates(subset= (new_groupby_cols))
    dataframe_columns = new_groupby_cols + [item for item in newly_created_cols if item not in new_groupby_cols]
    dataframe.sort_values(by = ['geography_name'], inplace = True)
    dataframe.fillna(' ', inplace=True)
    return dataframe[dataframe_columns]     

# Download button for the whole dataframe
# Function to write dataframes to different sheets in an Excel file
def to_excel(dfs, sheet_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for df, sheet_name in zip(dfs, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    # No need to call writer.save() as it is handled by the context manager
    processed_data = output.getvalue()
    return processed_data   




# #=============================================
# ### APP RUN
# #=============================================




partitions_query = """
create or replace table main.partitions as
select    *
    from read_parquet('az://$web/partition_df.parquet');
"""

partition_df = create_partitions_df(partitions_query)



st.sidebar.header("Select Filters")
st.sidebar.markdown("""Use the dropdowns below to reduce the size of the Census data.  
    First, specify whether to automatically select the "best" vintage or to include "all vintages" 
    if you need populations from non-standard vintage years (if you don't don't know, select "best").  
    Then, pick a geography level and the year(s) for which you want the population estimate.""")



selected_vintage_best, selected_population_year_str, selected_geography_level = partition_filter()


query = f"""
create or replace table main.filtered_data as
    select * 
    from read_parquet('az://$web/census_long.parquet/**/*.parquet', hive_partitioning = 1)
    WHERE vintage_best IN ({selected_vintage_best})
    AND CAST(geography_level as string) = ?
    AND CAST(population_year as string) IN ({selected_population_year_str})

"""   

try:
	conn.execute(query, [selected_geography_level]) 
except Exception as e:
    st.markdown("### ***Please select all the partitions to filter***")  



#=============================================
### Sidebar for filtering, pivoting and grouping
#=============================================
st.sidebar.header("Select Pivot and Group By Variables")
st.sidebar.markdown("""The selection features below are designed to streamline your desired output and orient the data in a digestible manner. 
    First, make the Demographic selections [Age, Race, Ethnicity, Sex] that you need. 
    If you do not want the data to be aggregated by a certain demographic, select the ‘NO [**insert demographic**] Breakdown’ option from the drop-down for that specific demographic option.""")

#=============================================
### Demographic Filters
#=============================================

try:
	age_query = f"""
	select distinct age_group from main.filtered_data
	order by age_group
	"""
	selected_age_string = age_selection(age_query)
except Exception as e:
    st.markdown("")


try: 
	race_query = f"""
	select distinct race from main.filtered_data
	where CAST(age_group as string) in ({selected_age_string})
	order by race"""
	selected_race_string = race_selection(race_query)
except Exception as e:
    st.markdown("### ***Please select age group to Filter***")


try:
	ethnicity_query = f"""
	select distinct ethnicity from main.filtered_data
	where CAST(race as string) in ({selected_race_string})
	order by ethnicity
	"""
	selected_ethnicity_string = ethnicity_selection(ethnicity_query)
except Exception as e:
    st.markdown("### ***Please select race to Filter***")	

try:
	sex_query = f"""
	select distinct gender from main.filtered_data
	where CAST(ethnicity as string) in ({selected_ethnicity_string})
	order by gender
	"""
	selected_sex_string = sex_seletion(sex_query)
except Exception as e:
    st.markdown("### ***Please select ethnicity to Filter***")	

try:
	additional_query = f"""
	create or replace table main.additionally_filtered_data as
	    select * from main.filtered_data
	    where CAST(race as string) IN ({selected_race_string})
	    AND CAST(age_group as string) IN ({selected_age_string})
	    AND CAST(ethnicity as string) in ({selected_ethnicity_string})
	    AND CAST(gender as string) in ({selected_sex_string})
	   ;
	"""

	conn.execute(additional_query)
except Exception as e:
    st.markdown("### ***Please select all Demographic Columns to Filter***")	



# #=============================================
# ### Pivot and Group by
# #=============================================
# # Multiselect for pivoting columns

# Describe query to get column names
try: 
    describe_query = "DESCRIBE main.filtered_data;"
    result = conn.execute(describe_query)

    # Extract column names from the result
    column_names = [row[0] for row in result.fetchall()]




    exclude_columns = ['source_id', 'vintage_id', 'vintage_type', 'geography_name', 'geography_id']
    column_names = [elem for elem in column_names if elem not in exclude_columns]
    st.sidebar.markdown("## Pivot Columns")
    pivot_columns = st.sidebar.multiselect("""Select the columns of the Demographic selections that you 
        desire to aggregate by (these are all of the selections you made above that are not filtered to the ‘NO [**insert demographic**] Breakdown’ selection)""", column_names)
    groupby_column_lst = [column for column in column_names if column not in pivot_columns]
    # Multiselect for grouping columns
    st.sidebar.markdown("## Group By Variables")
    group_by_columns = st.sidebar.multiselect("""Choose additional variables to group by (Optional). """, groupby_column_lst)
    group_by_columns.extend(exclude_columns)

    # Build the SQL query with parameterized query and dynamically include selected columns
    pivot_columns_str = ', '.join(pivot_columns)
    group_by_columns_str = ', '.join(group_by_columns)

    ### Pivot and Group by query
    query = f"""
    pivot main.additionally_filtered_data
        on {pivot_columns_str}
        using sum(population) 
        group by {group_by_columns_str};
"""
except Exception as e:
    st.markdown(" ") 

# Execute the SQL query with the selected values
try:
    data = conn.execute(query).df()
    # Identify decimal columns
    # Convert decimal columns to integers
    columns_to_convert = [col for col in data.columns if col not in groupby_column_lst]
    data.replace(to_replace=[None], value=0, inplace=True)
    data.sort_values(by= [a for a in data.columns], inplace = True)
    data.fillna(' ', inplace=True)
    data[columns_to_convert] = data[columns_to_convert].astype(str).applymap(lambda x: x.split('.')[0])
    data['geography_name'] = data['geography_name'].str.upper()



    new_created_columns = [item for item in data.columns if item not in group_by_columns]
                
    pop_sheet = create_pop_sheet(data, group_by_columns, new_created_columns)

    source_id_sheet = source_id_sheet(data, group_by_columns, new_created_columns)

    vintage_id_sheet = vintage_id_sheet(data, group_by_columns, new_created_columns)


    st.table(pop_sheet.head(10))
    st.write(pop_sheet.shape)



# Download button for the whole dataframe

    # Button to download the Excel file
    if st.button('Download Dataframes as Excel'):
        file = to_excel([pop_sheet, source_id_sheet, vintage_id_sheet], ['Population Value', 'Source_Id', 'Vintage_Id'])
        st.download_button(
            label="Download Excel",
            data=file,
            file_name="census_file.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



except Exception as e:
    st.markdown("### ***Please select columns to Pivot and Group by***")






