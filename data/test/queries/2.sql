select
  int_col,
  float_col,
  date_col
from
  test
where (
  int_col < 600
  and date_col between date '2020-05-01' and date '2020-12-31'
  and float_col > 0.9
) or (
  int_col < 600
  and date_col between date '2020-05-01' and date '2020-12-31'
  and float_col < 0.1
)