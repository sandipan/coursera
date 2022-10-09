function month = June2015()
  days = cellstr(['Mon'; 'Tue'; 'Wed'; 'Thu'; 'Fri'; 'Sat'; 'Sun']);
  month = {};
  for day=1:30
      month{day, 1} = 'June';
      month{day, 2} =  day;
      month{day, 3} = days{mod(day - 1, 7) + 1};
  end
end
