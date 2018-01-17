function [s, r, c] = mSample(m)
    s = size(m)
    r = m(round(rows(m)*rand),:);
    c = m(:,round(columns(m)*rand));
