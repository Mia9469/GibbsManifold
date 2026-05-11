%% remove_stars.m
%  Open fig2_NEW.fig, delete all pentagram (V2 star) scatter objects,
%  change all remaining markers to squares, and save as fig2_V1only.fig

fig = openfig('fig2_NEW.fig', 'visible');

% --- Step 1: delete pentagram (V2) scatter objects ---
sc_star = findall(fig, 'Type', 'Scatter', 'Marker', 'pentagram');
fprintf('Found %d pentagram scatter objects — deleting.\n', numel(sc_star));
delete(sc_star);

ln_star = findall(fig, 'Type', 'Line', 'Marker', 'pentagram');
if ~isempty(ln_star)
    fprintf('Found %d pentagram line objects — deleting.\n', numel(ln_star));
    delete(ln_star);
end

% --- Step 2: change all remaining markers to square ---
sc_all = findall(fig, 'Type', 'Scatter');
for i = 1:numel(sc_all)
    if ~strcmp(sc_all(i).Marker, 'none')
        sc_all(i).Marker = 'square';
    end
end
fprintf('Changed %d scatter objects to square marker.\n', numel(sc_all));

ln_all = findall(fig, 'Type', 'Line');
for i = 1:numel(ln_all)
    if ~strcmp(ln_all(i).Marker, 'none')
        ln_all(i).Marker = 'square';
    end
end

savefig(fig, 'fig2_V1only.fig');
fprintf('Saved as fig2_V1only.fig\n');
