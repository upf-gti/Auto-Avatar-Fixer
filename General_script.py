import bpy
import bmesh
import os
import json
import math
from math import sqrt
from mathutils import Matrix, Vector, Quaternion, Euler


#############################
#### Manual configurator ####
#############################
#Here you have to indicate the names of different parts of the body of the avatar in order to search for them
#Remember to separate the objects if they are joined together with multiple materials
#You can change the names of the object to anyting you want, but remember to change before putting them below 

hair_names = ["name_1", "name_2", "..."] #Input here all the names of the objects that are related to the hair
eye_main_names = ["ScleraMesh"] #Input here the names of the main eye names (spheres) normally sclera , if there are separated input both if it's one object input the name alone
eye_complementary_names = ["IrisMesh"] #Input here all the names of the objects that are related to the eyes e.g cornea, iris... (objects that are not a sphere)
shapekeys_names = ["HeadMesh", "EyeLashRightMesh", "EyeLashLeftMesh", "TongueMesh","UpperTeethMesh", "LowerTeethMesh"] #Input here all the names of the objects that need to keep their shapekeys (normally meshes around the head head or face, eyebrows, eyelashes, teeth, tongue)

############################
## Fix and merge armature ##
############################

def print(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")


def remove_bone_childs(armature, name):
    bpy.ops.object.mode_set(mode='EDIT',toggle=False)
    if name in armature.data.edit_bones:
        bone = armature.data.edit_bones[name]
        for children in bone.children:
            remove_bone_childs(armature, children.name)
        armature.data.edit_bones.remove(bone)

def apply_transforms(armature):
    bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action='DESELECT')
    
    
def select_armature(armature):
    bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT',toggle=False)
    bpy.ops.armature.select_all(action='DESELECT')

def remove_bones_whitelist( armature , whitelist):
    bpy.ops.object.mode_set(mode='EDIT',toggle=False)
    for bone in armature.data.edit_bones:
        is_whitelisted = False
        for substring in whitelist:
            if substring in bone.name and "Toe" not in bone.name:
                is_whitelisted = True
                break
        if not is_whitelisted:
            armature.data.edit_bones.remove(bone)

def remove_meshes(armature):
    for mesh in armature.children:
        bpy.data.objects.remove( mesh )
    
def select_object(obj):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

#########################
## Transfer shapekeys  ##
#########################

def copy_all_shape_keys(src, target):
    source = src
    dest = target
    bpy.ops.object.select_all(action='DESELECT')
    select_object(source)
    select_object(target)
    for v in bpy.context.selected_objects:
        if v is not dest:
            source = v
            break

    if source.data.shape_keys is None:
        print("Source object has no shape keys!") 
    else:
        for idx in range(1, len(source.data.shape_keys.key_blocks)):
            source.active_shape_key_index = idx
            bpy.ops.object.shape_key_transfer()
    bpy.context.object.show_only_shape_key = False

def transfer_shapekeys(armature):
    for obj in armature.children:
        if obj.data.shape_keys: #Already has shapekeys from mixamo
            continue
        if bpy.data.objects[obj.name+".001"].data.shape_keys:
            src = bpy.data.objects[obj.name+".001"]
            target = bpy.data.objects[obj.name]
            copy_all_shape_keys(src, target)


#Mixamorig skeleton cleanup
armature_mixamo = bpy.data.objects["Armature"]
armature_mixamo.select_set(True)
bpy.context.view_layer.objects.active = armature_mixamo
apply_transforms(armature_mixamo)

#armature_CC4 = bpy.data.objects["Armature.001"]
#apply_transforms(armature_CC4)

#set all obj to visible
for obj in bpy.data.objects:
    obj.hide_set(False)          
    obj.select_set(True)

bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

transfer_shapekeys(armature_mixamo)
#remove_meshes(armature_CC4)
for obj in bpy.data.objects:
    if obj.parent != armature_mixamo and obj != armature_mixamo:
        print(obj)
        print(obj.parent)
        bpy.data.objects.remove( obj )

bpy.ops.object.select_all(action='DESELECT')
armature_mixamo.select_set(True)
bpy.context.view_layer.objects.active = armature_mixamo
bpy.ops.object.mode_set(mode='OBJECT',toggle=False)

def select_object_name(object_name):
    ob = bpy.context.scene.objects[object_name]  # Get the object
    bpy.context.view_layer.objects.active = ob   # Make the cube the active object 
    ob.select_set(True)                          # Select the cube
    return ob

def find_related_word(input_word, word_dict):
    return word_dict.get(input_word, input_word)

def remove_substring(main_string, substring_to_remove,dict):
    w = main_string.replace(substring_to_remove, '')
    return find_related_word(w,bone_dict)

def dist3d(p1,p2):
    return sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

def final_pos(p1, vec, dist):
    vec_unit = [vec[0] / dist, vec[1] / dist, vec[2] / dist]
    a = [p1[0] + dist * vec_unit[0], p1[1] + dist * vec_unit[1], p1[2] + dist * vec_unit[2]]
    return a

def point_displacement(point, vec, disp):
    unit_vec = vec / np.linalg.norm(vec)
    return point + disp * unit_vec

def vecC(p1,p2):
    return [p2.x -p1.x, p2.y - p1.y, p2.z - p1.z]

bpy.ops.object.mode_set(mode='OBJECT',toggle=False)

#######################
## Correct materials ##
#######################
hair = None
brow = None
for object in bpy.data.objects:        
    if hasattr(object.data, 'materials'):
        for i,mat in enumerate(object.data.materials):
            if ".001" in mat.name: continue
            if "Hair" in mat.name:
                hair = object #we do this to get the hair object
            if "Brow" in mat.name:
                brow = object #we do this to get the hair object
            premat = bpy.data.materials.get(mat.name+".001")
            object.data.materials[i] = premat

for material in bpy.data.materials:
    if not ".001" in material.name:
        premat = bpy.data.materials.get(material.name+".001")
        if not premat is None:
            bpy.data.materials.remove(material)
            premat.name = premat.name[:-4]



#############
## Rigging ##
#############

def remove_vc(object):
    for i,vc in enumerate(object.vertex_groups):
        object.vertex_groups.remove(vc)

def merge_vertices(object):
    select_object(obj_copy)
    bpy.ops.object.mode_set(mode='EDIT',toggle=False)
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles()

def apply_autorig(obj, armature):
    bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
    remove_vc(obj)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')


def transfer_weights(source, target):
    remove_vc(target)
    bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
    bpy.ops.object.select_all(action='DESELECT')
    source.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.paint.weight_paint_toggle()
    bpy.ops.object.data_transfer(use_reverse_transfer=True, data_type='VGROUP_WEIGHTS', layers_select_src='NAME', layers_select_dst='ALL')
    bpy.ops.object.mode_set(mode='OBJECT',toggle=False)

def duplicate(obj, data=True, actions=True, collection=None):
    obj_copy = obj.copy()
    if data:
        obj_copy.data = obj_copy.data.copy()
    if actions and obj_copy.animation_data:
        obj_copy.animation_data.action = obj_copy.animation_data.action.copy()
    bpy.context.collection.objects.link(obj_copy)
    return obj_copy           

########################
## Reduce shapekeys ##
########################

def remove_all_shapekeys(obj):
    if hasattr(obj.data.shape_keys, 'key_blocks'): #Some dont have key_blocks properties
        for shapekey in obj.data.shape_keys.key_blocks:
            obj.shape_key_remove(shapekey)


#Rename the objects
eyes= None
for object in bpy.data.objects:
    if object.name not in shapekeys_names and object != armature_mixamo:
        #Only keep desired shapekeys
        remove_all_shapekeys(object)
bpy.ops.object.mode_set(mode='OBJECT',toggle=False)


#################################
## Eye bones & rename armature ##
#################################

def find_bone(bones, name):
    found_bone = [bone for bone in bones if bone.name == name]
    return found_bone
    
#Function to select all vertices from an object that are positive or negative in an axis
def select_object_vertices(obj, coord, positive):
    bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
    select_object(obj)    
    bpy.ops.object.mode_set(mode='EDIT',toggle=False)
    bpy.ops.mesh.select_all(action='DESELECT')
    
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    vertices= [e for e in bm.verts]
    verts_coord = [vert.co for vert in bm.verts]
    plain_verts = [vert.to_tuple() for vert in verts_coord]
    
    for vert in vertices:
        if plain_verts[vert.index][coord] >= 0 and positive:
            vert.select = True
        elif plain_verts[vert.index][coord] <= 0 and not positive:
            vert.select = True
        else:
            vert.select = False
  
#Bone rename dictionary
bone_dict = {
    "Spine": "Spine1",
    "Spine1": "Spine2",
    "Spine2": "Spine3",
    "HeadTop_End": "HeadEndSite",
    "RightEye": "RightEyeEndSite",
    "LeftEye": "LeftEyeEndSite",
    "LeftHandThumb4": "LeftHandThumbEndSite",
    "LeftHandIndex4": "LeftHandIndexEndSite",
    "LeftHandMiddle4": "LeftHandMiddleEndSite",
    "LeftHandRing4": "LeftHandRingEndSite",
    "LeftHandPinky4": "LeftHandPinkyEndSite",
    "RightHandThumb4": "RightHandThumbEndSite",
    "RightHandIndex4": "RightHandIndexEndSite",
    "RightHandMiddle4": "RightHandMiddleEndSite",
    "RightHandRing4": "RightHandRingEndSite",
    "RightHandPinky4": "RightHandPinkyEndSite",
    "LeftToeBase": "LeftToeBaseEndSite",
    "RightToeBase": "RightToeBaseEndSite",    
}

#Rename loop
for armature in bpy.data.armatures:
    armature.name = 'Armature'
    main_armature = armature
    for bone in armature.bones:
        #Sometimes mixamos rig will be mixamorig_ or mixamorig: depending on how old the rig is
        bone.name = remove_substring(bone.name,"mixamorig_", bone_dict)
        bone.name = remove_substring(bone.name,"mixamorig:", bone_dict)
        
#Create eye bones and vertex groups
bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
bpy.ops.object.select_all(action='DESELECT')

for eye_name in eye_main_names:# this for is in case the eyes are separated into 2 different objects
    eye_obj = bpy.data.objects[eye_name]
    select_object(eye_obj)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')

    loc = eye_obj.location
    y_disp = loc.y
    
    remove_vc(eye_obj)
    #Names for the eye bones, the mesh name has to be the same in order to work
    eyes = ["LeftEyeEndSite", "RightEyeEndSite"] 
    for eye in eyes:
        
        select_object_vertices(eye_obj, 0, eye == "LeftEyeEndSite")

        #Skip an eye if they're separated
        obj=bpy.context.object   
        if obj.mode == 'EDIT':
            bm=bmesh.from_edit_mesh(obj.data)
            res = any(v.select for v in bm.verts)
        if not res:
            continue
        
        #Add vertex group
        bpy.context.object.vertex_groups.new(name=eye)
        bpy.ops.object.vertex_group_assign()
        
        #Place 3D cursor in the eye center
        bpy.context.area.type = 'VIEW_3D'
        bpy.ops.view3d.snap_cursor_to_selected()
        bpy.context.area.type = 'TEXT_EDITOR'
        
        #Set the head bone as parent and create the bone
        bpy.ops.object.mode_set(mode='EDIT',toggle=True)
        obArm = select_object_name("Armature")
        bpy.ops.object.mode_set(mode='EDIT',toggle=True)
        ebs = obArm.data.edit_bones
        parent = ebs["Head"]
        eb = ebs.new(eye)
        eb.parent = parent
        #if the head and tail are the same, the bone is deleted
        eb.head = (bpy.context.scene.cursor.location.x, y_disp, bpy.context.scene.cursor.location.z )
        eb.tail = (bpy.context.scene.cursor.location.x, y_disp, bpy.context.scene.cursor.location.z +.05)
        
        #Reset the context so no object remains selected
        bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
        bpy.ops.object.select_all(action='DESELECT')
    
    #Place 3D cursor in the eye center
    bpy.context.area.type = 'VIEW_3D'
    bpy.ops.view3d.snap_cursor_to_center()
    bpy.context.area.type = 'TEXT_EDITOR'
    select_object(eye_obj)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')#'''

#This rigs the additional eye objects if there are some
bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
for eye_complements in eye_complementary_names:
    eye_obj = bpy.data.objects[eye_complements]
    remove_vc(eye_obj)
    apply_autorig(eye_obj, armature_mixamo)

#GOOD UNTIL HERE

###########################
## Fix skeleton position ##
###########################

-def quat_from_axis(xAxis, yAxis, zAxis):
    m00 = xAxis[0]
    m01 = xAxis[1]
    m02 = xAxis[2]
    m10 = yAxis[0]
    m11 = yAxis[1]
    m12 = yAxis[2]
    m20 = zAxis[0]
    m21 = zAxis[1]
    m22 = zAxis[2]
    t = m00 + m11 + m22
    if t > 0.0:
        s = sqrt(t + 1.0)
        w = s * 0.5 # |w| >= 0.5
        s = 0.5 / s
        x = (m12 - m21) * s
        y = (m20 - m02) * s
        z = (m01 - m10) * s
    elif m00 >= m11 and m00 >= m22:
        s = sqrt(1.0 + m00 - m11 - m22)
        x = 0.5 * s # |x| >= 0.5
        s = 0.5 / s
        y = (m01 + m10) * s
        z = (m02 + m20) * s
        w = (m12 - m21) * s
    elif m11 > m22:
        s = sqrt(1.0 + m11 - m00 - m22)
        y = 0.5 * s # |y| >= 0.5
        s = 0.5 / s
        x = (m10 + m01) * s
        z = (m21 + m12) * s
        w = (m20 - m02) * s
    else:
        s = sqrt(1.0 + m22 - m00 - m11)
        z = 0.5 * s # |z| >= 0.5
        s = 0.5 / s
        x = (m20 + m02) * s
        y = (m21 + m12) * s
        w = (m01 - m10) * s
    return Quaternion((w, x, y, z))

def mirrorQuat_X(quat):
    quat[2] = -quat[2] # neg Y
    quat[3] = -quat[3] # neg Z
    return quat

def create_rotation_constraint(bone, target_object, target_bone):
    if target_bone is None:
        return
    bpy.ops.object.mode_set(mode = 'POSE', toggle = True)
    bone.constraints.new(type='COPY_ROTATION')
    bpy.context.object.pose.bones[bone.name].constraints["Copy Rotation"].target = target_object
    bpy.context.object.pose.bones[bone.name].constraints["Copy Rotation"].subtarget = target_bone.name
    bpy.ops.object.mode_set(mode = 'POSE', toggle = True)
    
def create_rotation_constraint_recursive(pose_bone, target_object, target_bone):
    # Call create_rotation_constraint for the current pose bone
    create_rotation_constraint(pose_bone, target_object, target_bone)
    # Iterate through child bones
    for child in pose_bone.children:
        create_rotation_constraint_recursive(child, target_object, child.parent)

def set_bone_rolls_rec(bone):
    bone.roll = bone.parent.roll
    # Iterate through child bones
    for child in bone.children:
        set_bone_rolls_rec(child)

def set_bone_heads(bone):
    bone.head = bone.parent.tail
            
def align_direction(left_chain, parentIdx, right_chain, bones, bone_up):
    global old_L_location
    global old_R_location
    global new_L_location
    global new_R_location
    global head_L_location
    global head_R_location
    # compute first parent matrix
    obj_wm = bones[parentIdx].id_data.matrix_world
    pose_m = bones[parentIdx].matrix
    parent_wm = obj_wm @ pose_m
    parent_wq = parent_wm.to_quaternion()
    parent_wq.normalize()
    for idxL, idxR in zip(left_chain, right_chain):
        bone = bones[idxL]
        # Compute bone world rotation
        obj_wm = bones[idxL].id_data.matrix_world
        pose_m = bones[idxL].matrix
        child_wm = obj_wm @ pose_m # pose position
        child_wq = child_wm.to_quaternion()
        child_wq.normalize()
        parent_wm_inv = parent_wm.copy()
        parent_wm_inv.invert()
        parent_wq_inv = parent_wm_inv.to_quaternion()
        child_wm_rest = bones[idxL].bone.matrix_local # rest position
        child_lm_rest = parent_wm_inv @ child_wm_rest # local rest position of the bone
        child_lm = parent_wm_inv @ child_wm # local pose position of the bone
        child_lq = child_lm.to_quaternion()
        child_wm = parent_wm @ child_lm # child world rotation
        child_wq = child_wm.to_quaternion()
        child_wq.normalize()
        # Align and compute new bone rotation
        bone_fwrd = Vector((0.0, 0.0, 1.0))
        bone_fwrd.rotate(child_wq)
        bone_lft = bone_up.cross(bone_fwrd)
        bone_lft.normalize()
        bone_fwrd = bone_lft.cross(bone_up)
        bone_fwrd.normalize()
        child_q = quat_from_axis(bone_lft, bone_up, bone_fwrd)
        # Set the new rotation to the rest pose
        bpy.ops.object.mode_set(mode = 'EDIT')
        old_L_location = bones[idxL].tail.copy()
        old_R_location = bones[idxR].tail.copy()
        #matrix_to_modify = bones[idxL].matrix.copy()
        #matrix_to_modify = bpy.context.scene.collection.all_objects["Armature"].data.edit_bones[idxL].matrix.copy()
        matrix_to_modify = bones[idxL].matrix.copy()
        t = Matrix.Translation( matrix_to_modify.to_translation() )
        r = child_q.to_matrix().to_4x4() # (good one) pose_bone.bone.matrix_local.to_3x3().normalized().to_4x4()
        s = Matrix.Diagonal( matrix_to_modify.to_scale().to_4d() )
        m = t @ r @ s
        bone.matrix = m.copy()
        #bpy.context.scene.collection.all_objects["Armature"].data.edit_bones[idxL].matrix = m.copy()
        # Force child bone to start where the current bone ends
        if idxL is not left_chain[-1]:
            bpy.ops.object.mode_set(mode = 'EDIT')
            tail = bpy.context.scene.collection.all_objects["Armature"].data.edit_bones[idxL].tail.copy()
            bpy.context.scene.collection.all_objects["Armature"].data.edit_bones[idxL+1].head = tail    
        
        # Mirror for the right arm bones
        bone = bones[idxR]
        child_q_mirr = mirrorQuat_X(child_q)
        matrixR_to_modify = bones[idxR].matrix.copy()
        #matrixR_to_modify = bpy.context.scene.collection.all_objects["Armature"].data.edit_bones[idxR].matrix.copy()
        T = Matrix.Translation( matrixR_to_modify.to_translation() )
        R = child_q_mirr.to_matrix().to_4x4() 
        S = Matrix.Diagonal( matrixR_to_modify.to_scale().to_4d() )
        M = T @ R @ S
        #bpy.context.scene.collection.all_objects["Armature"].data.edit_bones[idxR].matrix = M.copy()
        bone.matrix = M.copy()
        # Force child bone to start where the current bone ends
            
        if idxR is not right_chain[-1]:
            tailR = bpy.context.scene.collection.all_objects["Armature"].data.edit_bones[idxR].tail.copy()
            bpy.context.scene.collection.all_objects["Armature"].data.edit_bones[idxR+1].head = tailR
        new_L_location = bones[idxL].tail.copy()
        new_R_location = bones[idxR].tail.copy()
        head_L_location = bones[idxL].head.copy()
        head_R_location = bones[idxR].head.copy()
        # On the next iteration, the current bone will be the parent
        parent_wm = m.copy()
        
    return

##########
# Main Script
##########

# get data structures needed   
bpy.context.view_layer.objects.active = bpy.data.objects.get("Armature")

# Calls to the functionT
align_direction( [10], 9, [34], bpy.context.object.pose.bones, Vector((1.0, 0.0, 0.0)) )

#Aligns child bones
armature = bpy.data.objects.get("Armature")
bpy.ops.object.mode_set(mode = 'EDIT')
set_bone_rolls_rec(bpy.data.objects["Armature"].data.edit_bones[12])
set_bone_rolls_rec(bpy.data.objects["Armature"].data.edit_bones[36])
bone_head_fix_list = [11, 12, 35, 36]
for bone in bone_head_fix_list:
    set_bone_heads(bpy.data.objects["Armature"].data.edit_bones[bone])

#Correct hand tail to match middle finger
#bpy.data.objects["Armature"].data.edit_bones[12].tail = bpy.data.objects["Armature"].data.edit_bones[21].head
#bpy.data.objects["Armature"].data.edit_bones[36].tail = bpy.data.objects["Armature"].data.edit_bones[45].head

bpy.ops.object.mode_set(mode = 'EDIT', toggle = True)

if armature and armature.type == 'ARMATURE':
    initial_bones = [10 ,34]
    for i in initial_bones:  
        root_pose_bone = armature.pose.bones[i]
        if root_pose_bone:
            create_rotation_constraint_recursive(root_pose_bone, armature, None)

#Apply pose
def duplicate_and_apply_armature_modifier():
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            armature_modifiers = [mod for mod in obj.modifiers if mod.type == 'ARMATURE']
            if armature_modifiers:
                if obj.data.shape_keys:
                    continue
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_copy(modifier="Armature")
                                                
                # Apply the first armature modifier
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_apply(modifier="Armature.001")

bpy.ops.object.mode_set(mode = 'OBJECT')
duplicate_and_apply_armature_modifier()
bpy.context.view_layer.objects.active = bpy.data.objects["Armature"]
bpy.ops.object.mode_set(mode = 'POSE')
bpy.context.object.data.pose_position = 'POSE'
bpy.ops.pose.armature_apply(selected=False)
bpy.ops.object.mode_set(mode = 'OBJECT')

#Remove the constraints on arm & hands bones
def remove_constraints_recursive(pose_bone):
    # Call create_rotation_constraint for the current pose bone
    for c in pose_bone.constraints:
        pose_bone.constraints.remove(c)
    # Iterate through child bones
    for child in pose_bone.children:
        remove_constraints_recursive(child)

armature =  bpy.data.objects["Armature"]
if armature and armature.type == 'ARMATURE':
    initial_bones = [10 ,34]
    for i in initial_bones:  
        root_pose_bone = armature.pose.bones[i]
        if root_pose_bone:
            remove_constraints_recursive(root_pose_bone)

#We do this to avoid having any transformation on the POSE position
bpy.ops.object.mode_set(mode = 'OBJECT')
bpy.context.view_layer.objects.active = bpy.data.objects["Armature"]
bpy.ops.object.mode_set(mode = 'POSE')
bpy.context.object.data.pose_position = 'REST'
bpy.ops.pose.armature_apply(selected=False)
bpy.ops.object.mode_set(mode = 'OBJECT')

#Rotate the tumbs by a fixed angle to prevent mesh overlapping
def rotate_bone(bone, deg):
    
    angle_radians = math.radians(deg)
            
    # Create a quaternion for rotation
    rotation_quaternion = Quaternion((0, 0, 1), angle_radians)

    # Apply the rotation to the bone
    bone.rotation_quaternion = rotation_quaternion @ bone.rotation_quaternion
    
bpy.ops.object.mode_set(mode = 'POSE')    
bpy.context.object.data.pose_position = 'POSE'
#Move the thumbs 60º
rotate_bone(armature.pose.bones["LeftHandThumb1"],30)
rotate_bone(armature.pose.bones["RightHandThumb1"],-30)

#Apply the pose as bind pose
bpy.ops.object.mode_set(mode = 'OBJECT')
duplicate_and_apply_armature_modifier()
bpy.context.view_layer.objects.active = bpy.data.objects["Armature"]
bpy.ops.object.mode_set(mode = 'POSE')
bpy.context.object.data.pose_position = 'POSE'
bpy.ops.pose.armature_apply(selected=False)
bpy.ops.object.mode_set(mode = 'OBJECT')#We do this to avoid having any transformation on the POSE position
bpy.ops.object.mode_set(mode = 'OBJECT')
bpy.context.view_layer.objects.active = bpy.data.objects["Armature"]
bpy.ops.object.mode_set(mode = 'POSE')
bpy.context.object.data.pose_position = 'REST'
bpy.ops.pose.armature_apply(selected=False)
bpy.ops.object.mode_set(mode = 'OBJECT')

#Rotate the tumbs by a fixed angle to prevent mesh overlapping
def rotate_bone(bone, deg):
    
    angle_radians = math.radians(deg)
            
    # Create a quaternion for rotation
    rotation_quaternion = Quaternion((0, 0, 1), angle_radians)

    # Apply the rotation to the bone
    bone.rotation_quaternion = rotation_quaternion @ bone.rotation_quaternion
    
bpy.ops.object.mode_set(mode = 'POSE')    
bpy.context.object.data.pose_position = 'POSE'
#Move the thumbs 60º
rotate_bone(armature.pose.bones[29],30)
rotate_bone(armature.pose.bones[45],-30)

#Apply the pose as bind pose
bpy.ops.object.mode_set(mode = 'OBJECT')
duplicate_and_apply_armature_modifier()
bpy.context.view_layer.objects.active = bpy.data.objects["Armature"]
bpy.ops.object.mode_set(mode = 'POSE')
bpy.context.object.data.pose_position = 'POSE'
bpy.ops.pose.armature_apply(selected=False)
bpy.ops.object.mode_set(mode = 'OBJECT')