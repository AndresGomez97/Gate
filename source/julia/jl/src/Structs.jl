
##################################################################################################
######################################## Structs #################################################
##################################################################################################

# Int3 struct
struct Int3
    x::Int32
    y::Int32
    z::Int32
end

# Float3 struct
struct Float3
    x::Float32
    y::Float32
    z::Float32
end

# Cuda StackParticle struct
struct CuStackParticle
	E::CuPtr{Float32}
	dx::CuPtr{Float32}
    dy::CuPtr{Float32}
    dz::CuPtr{Float32}
    px::CuPtr{Float32}
    py::CuPtr{Float32}
    pz::CuPtr{Float32}
    t::CuPtr{Float32}
    type::CuPtr{UInt16}
    eventID::CuPtr{UInt32}
    trackID::CuPtr{UInt32}
    seed::CuPtr{UInt32}
    active::CuPtr{Char}
	endsimu::CuPtr{Char}
	table_x_brent::CuPtr{UInt64}
    size::UInt32
end

# Julia StackParticle struct
struct JlStackParticle
	E::Array{Float32}
	dx::Array{Float32}
    dy::Array{Float32}
    dz::Array{Float32}
    px::Array{Float32}
    py::Array{Float32}
    pz::Array{Float32}
    t::Array{Float32}
    type::Array{UInt16}
    eventID::Array{UInt32}
    trackID::Array{UInt32}
    seed::Array{UInt32}
    active::Array{Char}
    endsimu::Array{Char}
    table_x_brent::Array{UInt64}
    size::UInt32
end

# Julia StackParticle with CuArrays struct
struct JlCuStackParticle
	d_E::CuArray{Float32}
	d_dx::CuArray{Float32}
    d_dy::CuArray{Float32}
    d_dz::CuArray{Float32}
    d_px::CuArray{Float32}
    d_py::CuArray{Float32}
    d_pz::CuArray{Float32}
    d_t::CuArray{Float32}
    d_type::CuArray{UInt16}
    d_eventID::CuArray{UInt32}
    d_trackID::CuArray{UInt32}
    d_seed::CuArray{UInt32}
    d_active::CuArray{Char}
    d_endsimu::CuArray{Char}
    d_table_x_brent::CuArray{UInt64}
    size::UInt32
end

# Cuda Activities struct
struct CuActivities
    nb_activities::UInt32
    tot_activity::Float32
    act_index::CuPtr{UInt32}
    act_cdf::CuPtr{Float32}
end

# Julia Activities struct
struct JlActivities
    nb_activities::UInt32
    tot_activity::Float32
    act_index::Array{UInt32}
    act_cdf::Array{Float32}
end

#  Julia Activities with CuArrays struct
struct JlCuActivities
    nb_activities::UInt32
    tot_activity::Float32
    d_act_index::CuArray{UInt32}
    d_act_cdf::CuArray{Float32}
end

# Julia Volume struct
struct JlVolume
    data::Array{UInt16}
    mem_data::UInt32
    size_in_mm::Float3
    size_in_vox::Int3
    voxel_size::Float3
    nb_voxel_volume::Int32
    nb_voxel_slice::Int32
    position::Float3
end

# Cuda Volume struct
struct CuVolume
    data::CuPtr{UInt16}
    mem_data::UInt32
    size_in_mm::Float3
    size_in_vox::Int3
    voxel_size::Float3
    nb_voxel_volume::Int32
    nb_voxel_slice::Int32
    position::Float3
end

# Julia Volume with CuArray struct
struct JlCuVolume
    data::CuArray{UInt16}
    mem_data::UInt32
    size_in_mm::Float3
    size_in_vox::Int3
    voxel_size::Float3
    nb_voxel_volume::Int32
    nb_voxel_slice::Int32
    position::Float3
end

# Julia Materials struct
struct JlMaterials
    nb_materials::UInt32
    nb_elements_total::UInt32
    nb_elements::Array{UInt16}
    index::Array{UInt16}
    mixture::Array{UInt16}
    atom_num_dens::Array{Float32}
    nb_atoms_per_vol::Array{Float32}
    nb_electrons_per_vol::Array{Float32}
    electron_cut_energy::Array{Float32}
    electron_max_energy::Array{Float32}
    electron_mean_excitation_energy::Array{Float32}
    rad_length::Array{Float32}
    fX0::Array{Float32}
    fX1::Array{Float32}
    fD0::Array{Float32}
    fC::Array{Float32}
    fA::Array{Float32}
    fM::Array{Float32}
end

# Julia Materials struct
struct CuMaterials
    nb_materials::UInt32
    nb_elements_total::UInt32
    nb_elements::CuPtr{UInt16}
    index::CuPtr{UInt16}
    mixture::CuPtr{UInt16}
    atom_num_dens::CuPtr{Float32}
    nb_atoms_per_vol::CuPtr{Float32}
    nb_electrons_per_vol::CuPtr{Float32}
    electron_cut_energy::CuPtr{Float32}
    electron_max_energy::CuPtr{Float32}
    electron_mean_excitation_energy::CuPtr{Float32}
    rad_length::CuPtr{Float32}
    fX0::CuPtr{Float32}
    fX1::CuPtr{Float32}
    fD0::CuPtr{Float32}
    fC::CuPtr{Float32}
    fA::CuPtr{Float32}
    fM::CuPtr{Float32}
end

# Julia Materials struct
struct JlCuMaterials
    nb_materials::UInt32
    nb_elements_total::UInt32
    nb_elements::CuArray{UInt16}
    index::CuArray{UInt16}
    mixture::CuArray{UInt16}
    atom_num_dens::CuArray{Float32}
    nb_atoms_per_vol::CuArray{Float32}
    nb_electrons_per_vol::CuArray{Float32}
    electron_cut_energy::CuArray{Float32}
    electron_max_energy::CuArray{Float32}
    electron_mean_excitation_energy::CuArray{Float32}
    rad_length::CuArray{Float32}
    fX0::CuArray{Float32}
    fX1::CuArray{Float32}
    fD0::CuArray{Float32}
    fC::CuArray{Float32}
    fA::CuArray{Float32}
    fM::CuArray{Float32}
end

# Julia CoordHex2 struct
struct JlCoordHex2
    y::Array{Float64}
  	z::Array{Float64}
  	size::UInt32
end

# Cuda CoordHex2 struct
struct CuCoordHex2
    y::CuPtr{Float64}
  	z::CuPtr{Float64}
  	size::UInt32
end

# Julia CoordHex2 with CuArray struct
struct JlCuCoordHex2
    y::CuArray{Float64}
  	z::CuArray{Float64}
  	size::UInt32
end

# Julia Dosimetry struct
struct JlDosimetry
    edep::Array{Float32}
    edep2::Array{Float32}
    mem_data::UInt32
    size_in_mm::Float3
    size_in_vox::Int3
    voxel_size::Float3
    nb_voxel_volume::Int32
    nb_voxel_slice::Int32
    position::Float3
end

# Julia Dosimetry struct
struct CuDosimetry
    edep::CuPtr{Float32}
    edep2::CuPtr{Float32}
    mem_data::UInt32
    size_in_mm::Float3
    size_in_vox::Int3
    voxel_size::Float3
    nb_voxel_volume::Int32
    nb_voxel_slice::Int32
    position::Float3
end

# Julia Dosimetry with CuArray struct
struct JlCuDosimetry
    edep::CuArray{Float32}
    edep2::CuArray{Float32}
    mem_data::UInt32
    size_in_mm::Float3
    size_in_vox::Int3
    voxel_size::Float3
    nb_voxel_volume::Int32
    nb_voxel_slice::Int32
    position::Float3
end

# Colli struct
struct Colli
    size_x::Int32
    size_y::Int32
    size_z::Int32
    HexaRadius::Float64
    HexaHeight::Float64
    CubRepNumY::Int32
    CubRepNumZ::Int32
    CubRepVecX::Float64
    CubRepVecY::Float64
    CubRepVecZ::Float64
    LinRepVecX::Float64
    LinRepVecY::Float64
    LinRepVecZ::Float64
end

##################################################################################################
######################################## Convertions #############################################
##################################################################################################

# JlStackParticle -> JlCuStackParticle
Base.convert(::Type{JlCuStackParticle},jlStackparticle::JlStackParticle) = JlCuStackParticle(CuArray(jlStackparticle.E), CuArray(jlStackparticle.dx), CuArray(jlStackparticle.dy), CuArray(jlStackparticle.dz), CuArray(jlStackparticle.px), CuArray(jlStackparticle.py), CuArray(jlStackparticle.pz), CuArray(jlStackparticle.t), CuArray(jlStackparticle.type), CuArray(jlStackparticle.eventID), CuArray(jlStackparticle.trackID), CuArray(jlStackparticle.seed), CuArray(jlStackparticle.active), CuArray(jlStackparticle.endsimu), CuArray(jlStackparticle.table_x_brent), jlStackparticle.size)

# JlCuStackParticle -> CuStackParticle
Base.cconvert(::Type{CuStackParticle}, jlCuStackParticle::JlCuStackParticle) = CuStackParticle(pointer(jlCuStackParticle.d_E), pointer(jlCuStackParticle.d_dx), pointer(jlCuStackParticle.d_dy), pointer(jlCuStackParticle.d_dz), pointer(jlCuStackParticle.d_px), pointer(jlCuStackParticle.d_py), pointer(jlCuStackParticle.d_pz), pointer(jlCuStackParticle.d_t), pointer(jlCuStackParticle.d_type), pointer(jlCuStackParticle.d_eventID), pointer(jlCuStackParticle.d_trackID), pointer(jlCuStackParticle.d_seed), pointer(jlCuStackParticle.d_active), pointer(jlCuStackParticle.d_endsimu), pointer(jlCuStackParticle.d_table_x_brent), jlCuStackParticle.size)

# JlActivities -> JlCuActivities
Base.convert(::Type{JlCuActivities},jlActivity::JlActivities) = JlCuActivities(jlActivity.nb_activities, jlActivity.tot_activity, CuArray(jlActivity.act_index), CuArray(jlActivity.act_cdf))

# JlCuActivities -> CuActivities
Base.cconvert(::Type{CuActivities},jlCuActivity::JlCuActivities) = CuActivities(jlCuActivity.nb_activities, jlCuActivity.tot_activity, pointer(jlCuActivity.d_act_index), pointer(jlCuActivity.d_act_index))

# JlVolume -> JlCuVolume
Base.convert(::Type{JlCuVolume},jlVolume::JlVolume) = JlCuVolume(CuArray(jlVolume.data), jlVolume.mem_data, jlVolume.size_in_mm, jlVolume.size_in_vox, jlVolume.voxel_size, jlVolume.nb_voxel_volume, jlVolume.nb_voxel_slice, jlVolume.position)

# JlCuVolume -> CuVolume
Base.cconvert(::Type{CuVolume},jlCuVolume::JlCuVolume) = CuVolume(pointer(jlCuVolume.data), jlCuVolume.mem_data, jlCuVolume.size_in_mm, jlCuVolume.size_in_vox, jlCuVolume.voxel_size, jlCuVolume.nb_voxel_volume, jlCuVolume.nb_voxel_slice, jlCuVolume.position)

# JlMaterials -> JlCuMaterials
Base.convert(::Type{JlCuMaterials},jlMaterials::JlMaterials) = JlCuMaterials(jlMaterials.nb_materials, jlMaterials.nb_elements_total, CuArray(jlMaterials.nb_elements), CuArray(jlMaterials.index), CuArray(jlMaterials.mixture), CuArray(jlMaterials.atom_num_dens), CuArray(jlMaterials.nb_atoms_per_vol), CuArray(jlMaterials.nb_electrons_per_vol), CuArray(jlMaterials.electron_cut_energy), CuArray(jlMaterials.electron_max_energy), CuArray(jlMaterials.electron_mean_excitation_energy), CuArray(jlMaterials.rad_length), CuArray(jlMaterials.fX0), CuArray(jlMaterials.fX1), CuArray(jlMaterials.fD0), CuArray(jlMaterials.fC), CuArray(jlMaterials.fA), CuArray(jlMaterials.fM))

# JlCuMaterials -> CuMaterials
Base.cconvert(::Type{CuMaterials},jlCuMaterials::JlCuMaterials) = CuMaterials(jlCuMaterials.nb_materials, jlCuMaterials.nb_elements_total, pointer(jlCuMaterials.nb_elements), pointer(jlCuMaterials.index), pointer(jlCuMaterials.mixture), pointer(jlCuMaterials.atom_num_dens), pointer(jlCuMaterials.nb_atoms_per_vol), pointer(jlCuMaterials.nb_electrons_per_vol), pointer(jlCuMaterials.electron_cut_energy), pointer(jlCuMaterials.electron_max_energy), pointer(jlCuMaterials.electron_mean_excitation_energy), pointer(jlCuMaterials.rad_length), pointer(jlCuMaterials.fX0), pointer(jlCuMaterials.fX1), pointer(jlCuMaterials.fD0), pointer(jlCuMaterials.fC), pointer(jlCuMaterials.fA), pointer(jlCuMaterials.fM))

# JlCoordHex2 -> JlCuCoordHex2
Base.convert(::Type{JlCuCoordHex2},jlCoordHex2::JlCoordHex2) = JlCuCoordHex2(CuArray(jlCoordHex2.y), CuArray(jlCoordHex2.z), jlCoordHex2.size)

# JlCuCoordHex2 -> CuCoordHex2
Base.cconvert(::Type{CuCoordHex2},jlCuCoordHex2::JlCuCoordHex2) = CuCoordHex2(pointer(jlCuCoordHex2.y), pointer(jlCuCoordHex2.z), jlCuCoordHex2.size)

# JlDosimetry -> JlCuDosimetry
Base.convert(::Type{JlCuDosimetry},jlDosimetry::JlDosimetry) = JlCuDosimetry(CuArray(jlDosimetry.edep), CuArray(jlDosimetry.edep2), jlDosimetry.mem_data, jlDosimetry.size_in_mm, jlDosimetry.size_in_vox, jlDosimetry.voxel_size, jlDosimetry.nb_voxel_volume, jlDosimetry.nb_voxel_slice, jlDosimetry.position)

# JlCuDosimetry -> CuDosimetry
Base.cconvert(::Type{CuDosimetry},jlCuDosimetry::JlCuDosimetry) = CuDosimetry(pointer(jlCuDosimetry.edep), pointer(jlCuDosimetry.edep2), jlCuDosimetry.mem_data, jlCuDosimetry.size_in_mm, jlCuDosimetry.size_in_vox, jlCuDosimetry.voxel_size, jlCuDosimetry.nb_voxel_volume, jlCuDosimetry.nb_voxel_slice, jlCuDosimetry.position)
