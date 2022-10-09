function [time_min, dist_km] = light_time(dist_miles)
    dist_km = dist_miles * 1.609;
    time_min = dist_km / (300000 * 60);
end