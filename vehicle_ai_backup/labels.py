"""Danh mục tên xe Việt Nam dùng nhất quán với data/vehicles_vn.yaml.

Tên số ghế là NHÃN DO NGƯỜI GÁN; không có cách đếm chính xác số ghế
chỉ bằng hình chiếu bên ngoài của mọi chiếc xe.
"""

VN_LABELS = {
    # 2 banh/3 banh
    'bicycle': 'Xe đạp',
    'electric_bicycle': 'Xe đạp điện',
    'underbone_motorbike': 'Xe máy số',
    'scooter_motorbike': 'Xe tay ga',
    'electric_motorbike': 'Xe máy điện',
    'sport_motorbike': 'Xe mô tô thể thao',
    'cruiser_motorbike': 'Xe mô tô cruiser',
    'adventure_motorbike': 'Xe mô tô địa hình/du lịch',
    'three_wheel_motorbike': 'Xe ba bánh máy',
    'cargo_tricycle': 'Xe ba gác chở hàng',
    'cyclo': 'Xe xích lô',
    # xe con
    'sedan': 'Ô tô sedan',
    'hatchback': 'Ô tô hatchback',
    'coupe': 'Ô tô coupe',
    'convertible': 'Ô tô mui trần',
    'suv_5seat': 'SUV 5 chỗ (ước lượng)',
    'suv_7seat': 'SUV 7 chỗ (ước lượng)',
    'mpv_7seat': 'MPV 7 chỗ (ước lượng)',
    'pickup': 'Xe bán tải',
    'van_9seat': 'Xe van 9 chỗ (ước lượng)',
    'minibus_16seat': 'Xe khách 16 chỗ (ước lượng)',
    'minibus_29seat': 'Xe khách 29 chỗ (ước lượng)',
    'bus_32seat': 'Xe khách 32 chỗ (ước lượng)',
    'bus_45seat': 'Xe khách 45 chỗ (ước lượng)',
    'city_bus': 'Xe buýt đô thị',
    'coach_bus': 'Xe khách đường dài',
    # xe chuyen dung
    'ambulance': 'Xe cứu thương',
    'fire_truck': 'Xe cứu hỏa',
    'police_car': 'Xe cảnh sát',
    'school_bus': 'Xe đưa đón học sinh',
    'tow_truck': 'Xe cứu hộ giao thông',
    'garbage_truck': 'Xe chở rác',
    'road_sweeper': 'Xe quét đường',
    # xe tai
    'light_truck': 'Xe tải nhẹ',
    'medium_truck': 'Xe tải trung',
    'heavy_truck': 'Xe tải nặng',
    'dump_truck': 'Xe ben',
    'tanker_truck': 'Xe bồn chở chất lỏng',
    'refrigerated_truck': 'Xe tải đông lạnh',
    'container_truck': 'Xe đầu kéo container',
    'tractor_unit': 'Đầu kéo (không kéo container)',
    'flatbed_truck': 'Xe tải thùng lửng',
    'concrete_mixer': 'Xe trộn bê tông',
    'crane_truck': 'Xe tải gắn cẩu',
    # may cong trinh
    'forklift': 'Xe nâng hàng',
    'excavator': 'Máy xúc',
    'road_roller': 'Xe lu',
    'farm_tractor': 'Máy kéo nông nghiệp',
}

# COCO pretrained KHÔNG có các phân nhóm Việt Nam chi tiết.
COCO_BASELINE = {
    'bicycle': ('Xe đạp (chưa rõ loại)', 'bicycle'),
    'motorcycle': ('Xe máy / mô tô (chưa rõ loại)', 'motorcycle'),
    'car': ('Ô tô con (chưa rõ kiểu và số chỗ)', 'car'),
    'bus': ('Xe buýt / xe khách (chưa rõ số chỗ)', 'bus'),
    'truck': ('Xe tải (chưa rõ tải trọng/chức năng)', 'truck'),
}

# Danh mục gần đúng để chỉ cho phép ghép mask phân vùng tương thích với đối tượng.
BROAD_GROUPS = {
    'bicycle': 'bicycle', 'electric_bicycle': 'bicycle', 'cyclo': 'bicycle',
    'underbone_motorbike': 'motorcycle', 'scooter_motorbike': 'motorcycle',
    'electric_motorbike': 'motorcycle', 'sport_motorbike': 'motorcycle',
    'cruiser_motorbike': 'motorcycle', 'adventure_motorbike': 'motorcycle',
    'three_wheel_motorbike': 'motorcycle', 'cargo_tricycle': 'motorcycle',
    'sedan': 'car', 'hatchback': 'car', 'coupe': 'car', 'convertible': 'car',
    'suv_5seat': 'car', 'suv_7seat': 'car', 'mpv_7seat': 'car',
    'pickup': 'truck', 'van_9seat': 'car',
    'minibus_16seat': 'bus', 'minibus_29seat': 'bus', 'bus_32seat': 'bus',
    'bus_45seat': 'bus', 'city_bus': 'bus', 'coach_bus': 'bus',
    'school_bus': 'bus', 'ambulance': 'car', 'police_car': 'car',
    'fire_truck': 'truck', 'tow_truck': 'truck', 'garbage_truck': 'truck',
    'road_sweeper': 'truck', 'light_truck': 'truck', 'medium_truck': 'truck',
    'heavy_truck': 'truck', 'dump_truck': 'truck', 'tanker_truck': 'truck',
    'refrigerated_truck': 'truck', 'container_truck': 'truck',
    'tractor_unit': 'truck', 'flatbed_truck': 'truck', 'concrete_mixer': 'truck',
    'crane_truck': 'truck', 'forklift': None, 'excavator': None,
    'road_roller': None, 'farm_tractor': None,
}
