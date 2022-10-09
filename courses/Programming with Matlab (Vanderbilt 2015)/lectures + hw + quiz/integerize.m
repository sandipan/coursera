function int_class = integerize(A)
   int_class = 'NONE'
   val = max(max(A))
   if val < 2^8
       int_class = 'uint8'
   elseif val < 2^16
       int_class = 'uint16'
   elseif val < 2^32
       int_class = 'uint32'
   elseif val < 2^64
       int_class = 'uint64'
   end
end
