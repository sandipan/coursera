function month = May2015()
  days = cellstr(['Fri'; 'Sat'; 'Sun'; 'Mon'; 'Tue'; 'Wed'; 'Thu']);
  for day=1:31
      month(day).month = 'May';
      month(day).date = day;
      month(day).day = days{mod(day - 1, 7) + 1};
  end
end
