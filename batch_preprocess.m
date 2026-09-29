%% ========================================================================
%  多数据集 4DCT 数据预处理 (DIR-Lab + POPI 适配版)
%  功能: 读取 4DCT (DICOM/Raw) → 提取 2.5D 深度图 → Demons 配准 → 保存 .mat
%  输出: depth_maps (H, D, 10), DVFs (10个相位), pts_T00, pts_T50
% =========================================================================
clear; clc; close all;

main_path = 'D:/0临床科研/四维剂量重建/data/';

fprintf('==================================================\n');
fprintf('多数据集 4DCT 数据预处理 (DIR-Lab + POPI)\n');
fprintf('==================================================\n');

% ========================================================================
% 【1】DIR-Lab 数据集处理 (10例)
% ========================================================================
fprintf('\n【1】DIR-Lab 数据集处理\n');
fprintf('--------------------------------------------------\n');

for case_id = 1:10
    fprintf('检查 DIR-Lab Case %d / 10 ... ', case_id);
    
    case_path = fullfile(main_path, 'DIR-Lab', sprintf('Case%dPack', case_id));
    output_dir = fullfile(case_path, 'Processed');
    processed_file = fullfile(output_dir, sprintf('Case%d_DVF_Depth.mat', case_id));
    
    % 如果已经处理过，直接跳过
    if exist(processed_file, 'file')
        fprintf('✅ 已存在，跳过\n');
        continue;
    end
    
    if ~exist(case_path, 'dir')
        fprintf('⚠️ 目录不存在: %s，跳过\n', case_path);
        continue;
    end
    
    images_path = fullfile(case_path, 'Images');
    landmarks_path = fullfile(case_path, 'ExtremePhases');
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    
    [img_size, voxel_spacing] = get_dirlab_params(case_id);
    phases = {'00', '10', '20', '30', '40', '50', '60', '70', '80', '90'};
    
    % ---- 读取标志点 ----
    if case_id <= 5
        pts_T00 = load(fullfile(landmarks_path, sprintf('Case%d_300_T00_xyz.txt', case_id)));
        pts_T50 = load(fullfile(landmarks_path, sprintf('Case%d_300_T50_xyz.txt', case_id)));
    else
        pts_T00 = load(fullfile(landmarks_path, sprintf('case%d_dirLab300_T00_xyz.txt', case_id)));
        pts_T50 = load(fullfile(landmarks_path, sprintf('case%d_dirLab300_T50_xyz.txt', case_id)));
    end
    
    % ---- 读取图像与提取深度图 ----
    images_all = zeros([img_size, 10], 'int16');
    depth_maps = zeros(img_size(1), img_size(3), 10);
    
    for p = 1:10
        if case_id == 1, img_name = sprintf('case%d_T%s_s.img', case_id, phases{p});
        elseif case_id <= 5, img_name = sprintf('case%d_T%s-ssm.img', case_id, phases{p});
        else, img_name = sprintf('case%d_T%s.img', case_id, phases{p});
        end
        
        fid = fopen(fullfile(images_path, img_name), 'r');
        img = reshape(fread(fid, prod(img_size), 'int16'), img_size);
        fclose(fid);
        img = permute(img, [2, 1, 3]);
        images_all(:,:,:,p) = img;
        depth_maps(:,:,p) = extract_depth_map(img, voxel_spacing, img_size);
    end
    
    % ---- 配准计算 DVF ----
    DVFs = cell(10, 1);
    DVFs{1} = zeros([img_size, 3], 'single');
    ref_img = double(images_all(:,:,:,1));
    ref_img = (ref_img - min(ref_img(:))) / (max(ref_img(:)) - min(ref_img(:)) + 1e-6);
    
    for p = 2:10
        mov_img = double(images_all(:,:,:,p));
        mov_img = (mov_img - min(mov_img(:))) / (max(mov_img(:)) - min(mov_img(:)) + 1e-6);
        [DVF, ~] = imregdemons(mov_img, ref_img, [50 25 10], 'AccumulatedFieldSmoothing', 1.5, 'DisplayWaitbar', false);
        DVFs{p} = single(DVF);
    end
    
    save(processed_file, 'depth_maps', 'DVFs', 'pts_T00', 'pts_T50', '-v7.3');
    fprintf('  ✅ DIR-Lab Case %d 完成\n', case_id);
end

% ========================================================================
% 【2】POPI 数据集处理 (6例)
% ========================================================================
fprintf('\n【2】POPI 数据集处理\n');
fprintf('--------------------------------------------------\n');

phases = {'00', '10', '20', '30', '40', '50', '60', '70', '80', '90'};

for patient_id = 1:6
    % 支持 patient_01 或 patient_1 两种命名
    patient_folder_name = sprintf('patient_%02d', patient_id);
    patient_path = fullfile(main_path, 'POPI', patient_folder_name);
    
    if ~exist(patient_path, 'dir')
        patient_folder_name = sprintf('patient_%d', patient_id);
        patient_path = fullfile(main_path, 'POPI', patient_folder_name);
    end
    
    if ~exist(patient_path, 'dir')
        fprintf('⚠️ 找不到 POPI Patient %d 的文件夹 (%s)，跳过\n', patient_id, patient_path);
        continue;
    end
    
    output_dir = fullfile(patient_path, 'Processed');
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    
    processed_file = fullfile(output_dir, sprintf('Case%d_DVF_Depth.mat', patient_id));
    
    if exist(processed_file, 'file')
        fprintf('✅ POPI Patient %d 已存在，跳过\n', patient_id);
        continue;
    end
    
    fprintf('\n正在处理 POPI Patient %d (%s) ...\n', patient_id, patient_folder_name);
    
    % ---- 读取标志点 ----
    landmarks_path = fullfile(patient_path, 'landmarks');
    pts_T00 = load_popi_landmarks(landmarks_path, '00');
    pts_T50 = load_popi_landmarks(landmarks_path, '50');
    
    if isempty(pts_T00) || isempty(pts_T50)
        fprintf('  ❌ 标志点读取失败，请检查 %s\n', landmarks_path);
        continue;
    end
    fprintf('  标志点已加载: T00(%d个), T50(%d个)\n', size(pts_T00, 1), size(pts_T50, 1));
    
    % ---- 读取 3D DICOM 相位图像 ----
    fprintf('  读取 4DCT 相位 DICOM 图像...\n');
    images_all = [];
    voxel_spacing = [0.97, 0.97, 2.5]; % 默认初始化，读取DICOM后自动更新
    
    for p = 1:10
        phase_dir = fullfile(patient_path, phases{p});
        if ~exist(phase_dir, 'dir')
            fprintf('    ⚠️ 缺失相位文件夹: %s\n', phases{p});
            continue;
        end
        
        % 从 DICOM 文件夹中提取 3D 体数据
        [img_3d, vs_dicom] = read_dicom_phase_folder(phase_dir);
        
        if isempty(img_3d)
            fprintf('    ❌ 相位 %s DICOM 读取失败\n', phases{p});
            continue;
        end
        
        if p == 1
            voxel_spacing = vs_dicom;
            img_size = size(img_3d);
            images_all = zeros([img_size, 10], 'single');
            depth_maps = zeros(img_size(1), img_size(3), 10);
        end
        
        images_all(:,:,:,p) = single(img_3d);
        depth_maps(:,:,p) = extract_depth_map(img_3d, voxel_spacing, img_size);
        fprintf('    Phase %s: 深度图提取完成 [尺寸: %dx%dx%d]\n', phases{p}, img_size(1), img_size(2), img_size(3));
    end
    
    % ---- Demons 配准计算 DVF ----
    fprintf('  计算 3D 形变场 DVF (Demons 配准)...\n');
    DVFs = cell(10, 1);
    DVFs{1} = zeros([img_size, 3], 'single');
    
    ref_img = double(images_all(:,:,:,1));
    ref_norm = (ref_img - min(ref_img(:))) / (max(ref_img(:)) - min(ref_img(:)) + 1e-6);
    
    for p = 2:10
        fprintf('    配准 Phase %s -> Phase 00...\n', phases{p});
        mov_img = double(images_all(:,:,:,p));
        mov_norm = (mov_img - min(mov_img(:))) / (max(mov_img(:)) - min(mov_img(:)) + 1e-6);
        
        [DVF, ~] = imregdemons(mov_norm, ref_norm, [50 25 10], ...
            'AccumulatedFieldSmoothing', 1.5, 'DisplayWaitbar', false);
        DVFs{p} = single(DVF);
    end
    
    % ---- 保存结果 ----
    save(processed_file, 'depth_maps', 'DVFs', 'pts_T00', 'pts_T50', '-v7.3');
    fprintf('  ✅ 完成并保存: %s\n', processed_file);
end

fprintf('\n==================================================\n');
fprintf('🎉 所有数据集预处理完毕！\n');
fprintf('==================================================\n');


%% ========================================================================
%  辅助函数
%% ========================================================================

function [img_size, voxel_spacing] = get_dirlab_params(case_id)
    if case_id <= 5
        img_size = [256, 256, 94]; voxel_spacing = [0.97, 0.97, 2.5];
    else
        img_size = [512, 512, 128]; voxel_spacing = [0.97, 0.97, 2.5];
    end
    switch case_id
        case 1,  img_size(3) = 94;
        case 2,  img_size(3) = 112; voxel_spacing(1:2) = [1.16, 1.16];
        case 3,  img_size(3) = 104; voxel_spacing(1:2) = [1.15, 1.15];
        case 4,  img_size(3) = 99;  voxel_spacing(1:2) = [1.13, 1.13];
        case 5,  img_size(3) = 106; voxel_spacing(1:2) = [1.10, 1.10];
        case 7,  img_size(3) = 136;
        case 10, img_size(3) = 120;
    end
end

function [img_3d, voxel_spacing] = read_dicom_phase_folder(phase_dir)
    % 从 DICOM 文件夹高效读取 3D 图像序列
    img_3d = [];
    voxel_spacing = [0.97, 0.97, 2.5];
    
    try
        % 尝试优先使用 MATLAB 内置 Volume 读取器
        [V, spatial] = dicomreadVolume(phase_dir);
        img_3d = squeeze(V);
        if isinteger(img_3d)
            img_3d = single(img_3d);
        end
        % 自动提取体像素间距
        if ~isempty(spatial) && isprop(spatial, 'PixelSpacings')
            dy = spatial.PixelSpacings(1,1);
            dx = spatial.PixelSpacings(1,2);
            dz = 2.5;
            if isprop(spatial, 'PatientPositions') && size(spatial.PatientPositions, 1) > 1
                dz = abs(spatial.PatientPositions(2,3) - spatial.PatientPositions(1,3));
            end
            voxel_spacing = [dy, dx, dz];
        end
    catch
        % Fallback：传统逐张读取 DICOM
        dcm_files = dir(fullfile(phase_dir, '*.dcm'));
        if isempty(dcm_files)
            dcm_files = dir(fullfile(phase_dir, '*'));
            dcm_files = dcm_files(~[dcm_files.isdir]);
        end
        
        if isempty(dcm_files), return; end
        
        info = dicominfo(fullfile(phase_dir, dcm_files(1).name));
        H = double(info.Rows); W = double(info.Columns); D = length(dcm_files);
        img_3d = zeros(H, W, D, 'single');
        
        for z = 1:D
            img_3d(:,:,z) = single(dicomread(fullfile(phase_dir, dcm_files(z).name)));
        end
        
        if isfield(info, 'PixelSpacing')
            voxel_spacing(1:2) = info.PixelSpacing';
        end
        if isfield(info, 'SliceThickness')
            voxel_spacing(3) = info.SliceThickness;
        end
    end
end

function depth_map = extract_depth_map(img, voxel_spacing, img_size)
    H = img_size(1); W = img_size(2); D = img_size(3);
    img_norm = (double(img) - min(double(img(:)))) / (max(double(img(:))) - min(double(img(:))) + 1e-6);
    
    level = graythresh(img_norm(:, :, round(D/2)));
    body_mask = img_norm > level;
    for z = 1:D
        body_mask(:,:,z) = imfill(body_mask(:,:,z), 'holes');
    end
    
    CC = bwconncomp(body_mask, 26);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    if isempty(numPixels)
        depth_map = zeros(H, D);
        return;
    end
    [~, max_idx] = max(numPixels);
    patient_body = false(size(body_mask));
    patient_body(CC.PixelIdxList{max_idx}) = true;
    
    depth_map = NaN(H, D);
    for x = 1:H
        for z = 1:D
            y_skin = find(patient_body(x, :, z), 1, 'first');
            if ~isempty(y_skin)
                depth_map(x, z) = y_skin * voxel_spacing(2);
            end
        end
    end
    
    depth_map(1:round(H*0.2), :) = NaN;
    depth_map(round(H*0.8):end, :) = NaN;
    depth_map = fillmissing(depth_map, 'nearest');
    depth_map = medfilt2(depth_map, [5, 5]);
    depth_map = W * voxel_spacing(2) - depth_map;
end

function pts = load_popi_landmarks(landmarks_path, phase_str)
    % 自动搜索匹配包含 phase_str (如 '00' 或 '50') 的标志点文件
    pts = [];
    files = dir(fullfile(landmarks_path, ['*', phase_str, '*']));
    
    if isempty(files), return; end
    
    filepath = fullfile(landmarks_path, files(1).name);
    try
        % 兼容 .pts 与 .txt 格式
        raw = importdata(filepath);
        if isstruct(raw)
            data = raw.data;
        else
            data = raw;
        end
        % 如果第一行为数值点数，自动剔除
        if size(data, 2) == 1
            data = readmatrix(filepath, 'FileType', 'text');
        end
        if size(data, 2) > 3
            data = data(:, 1:3);
        end
        pts = data;
    catch
        pts = [];
    end
end