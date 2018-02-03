function [  ] = compare_files(filename1, filename2)

    data1 = load(filename1);

    data2 = load(filename2);

    fields = fieldnames(data1);

    for k = 1:numel(fields)
        differences = sum(sum(sum(sum(sum(sum(sum(data1.(fields{k})~=data2.(fields{k}))))))));
        fprintf('Field %s: %d (of %d)\n',fields{k}, differences, prod(size(data1.(fields{k}))));
    end


end

