function grade = letter_grade(score)
    grades = repmat('F', 1, 100);
    grades(91:end) = 'A';
    grades(81:90) = 'B';
    grades(71:80) = 'C';
    grades(61:70) = 'D';
    grade = grades(score)
end