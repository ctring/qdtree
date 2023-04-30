select
  int_col,
  float_col,
  date_col
from
  test
where
  int_col between 23 and 50
  and test.float_col > (select avg(float_col) from test)
  and date_col > date '2020-12-30'
order by
  int_col