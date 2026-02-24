bl_info = {
    "name": "Studio Batch Export",
    "author": "Joel",
    "version": (1, 0, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Export",
    "description": "Batch export tool for AST-based asset workflow",
    "category": "Import-Export",
}

import bpy
import os
import subprocess
import math
from pathlib import Path
from mathutils import Vector
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import StringProperty, BoolProperty, IntProperty, CollectionProperty, EnumProperty
import gpu
import blf
from bpy_extras import view3d_utils
from typing import List, Tuple, Set, Optional, Dict, Any


class ASTConstants:
    """Centralized constants for the AST workflow."""
    PREFIX = "AST_"
    COLLISION_PREFIX = "COL_"
    GRID_SPACING = 10.0
    POST_EXPORT_FRAMES = 5


# PropertyGroup for export items in the status list
class AST_ExportItemPG(PropertyGroup):
    name: StringProperty()
    status: StringProperty(default="pending")
    stage: StringProperty(default="")


def ensure_ast_properties(ast_object: bpy.types.Object) -> None:
    """
    Ensure an AST empty has required custom properties with default values.
    
    Args:
        ast_object: The Blender object to initialize properties on.
    """
    if "asset_name" not in ast_object:
        ast_object["asset_name"] = ""
    if "export_path" not in ast_object:
        ast_object["export_path"] = ""
    if "ast_active" not in ast_object:
        ast_object["ast_active"] = True
    if "ast_expanded" not in ast_object:
        ast_object["ast_expanded"] = True
    if "asset_group" not in ast_object:
        ast_object["asset_group"] = ""


def validate_ast_fields(ast_object: bpy.types.Object) -> Tuple[bool, Optional[str]]:
    """
    Simple field validation for an AST object.
    
    Args:
        ast_object: The AST object to validate.
        
    Returns:
        A tuple of (is_valid, error_message).
    """
    if not ast_object.get("asset_name", "").strip():
        return False, "Asset name is empty"
    if not ast_object.get("export_path", "").strip():
        return False, "Export path is empty"
    return True, None


def get_ast_children(ast_object: bpy.types.Object, recursive: bool = True) -> List[bpy.types.Object]:
    """
    Get all children of an AST object.
    
    Args:
        ast_object: The parent object to start from.
        recursive: Whether to include nested children.
        
    Returns:
        A list of child Blender objects.
    """
    children = []
    def collect_children(obj: bpy.types.Object):
        for child in obj.children:
            children.append(child)
            if recursive:
                collect_children(child)
    collect_children(ast_object)
    return children


def find_parent_ast(obj: bpy.types.Object) -> Optional[bpy.types.Object]:
    """
    Find the parent AST empty for a given object.
    
    Args:
        obj: The object to search upwards from.
        
    Returns:
        The parent AST object if found, otherwise None.
    """
    current = obj
    while current:
        if current.type == 'EMPTY' and current.name.startswith(ASTConstants.PREFIX):
            return current
        current = current.parent
    return None


def get_selected_ast_groups(context: bpy.types.Context) -> Set[int]:
    """
    Helper to get set of group indices for selected ASTs.
    
    Args:
        context: The current Blender context.
        
    Returns:
        A set of group index integers.
    """
    active_groups = set()
    for obj in context.selected_objects:
        if obj.type == 'EMPTY' and obj.name.startswith(ASTConstants.PREFIX):
            grp = obj.ast_group_index
            if grp != -1:
                active_groups.add(grp)
        else:
            parent = find_parent_ast(obj)
            if parent:
                grp = parent.ast_group_index
                if grp != -1:
                    active_groups.add(grp)
    return active_groups


def get_all_asts(context: bpy.types.Context) -> List[bpy.types.Object]:
    """
    Get all AST empties in the scene, sorted alphabetically.
    
    Args:
        context: The current Blender context.
        
    Returns:
        List of AST objects.
    """
    return sorted(
        [obj for obj in context.scene.objects 
         if obj.type == 'EMPTY' and obj.name.startswith(ASTConstants.PREFIX)],
        key=lambda x: x.name.lower()
    )


def get_selected_asts(context: bpy.types.Context) -> Set[bpy.types.Object]:
    """
    Get the set of AST objects based on current selection.
    Includes directly selected ASTs and parents of selected objects.
    
    Args:
        context: The current Blender context.
        
    Returns:
        A set of AST objects.
    """
    selected_asts = set()
    for obj in context.selected_objects:
        if obj.type == 'EMPTY' and obj.name.startswith(ASTConstants.PREFIX):
            selected_asts.add(obj)
        else:
            parent_ast = find_parent_ast(obj)
            if parent_ast:
                selected_asts.add(parent_ast)
    return selected_asts


def get_ast_stats(children: List[bpy.types.Object]) -> Dict[str, Any]:
    """
    Calculate 3D statistics (tris, mats, collisions) for a set of AST children.
    
    Args:
        children: List of child objects belonging to an AST.
        
    Returns:
        Dictionary containing counts and statistics.
    """
    stats = {
        'child_count': len(children),
        'tri_count': 0,
        'material_count': 0,
        'has_collision': False,
        'collision_count': 0,
        'collision_tri_count': 0,
    }
    
    materials = set()
    
    for child in children:
        if child.type != 'MESH' or not child.data:
            continue
        
        mesh = child.data
        is_collision = child.name.startswith(ASTConstants.COLLISION_PREFIX)
        
        # Calculate triangle count
        tri_count = sum(len(poly.vertices) - 2 for poly in mesh.polygons)
        
        if is_collision:
            stats['has_collision'] = True
            stats['collision_count'] += 1
            stats['collision_tri_count'] += tri_count
        else:
            stats['tri_count'] += tri_count
            # Collect materials (excluding collision)
            for mat_slot in child.material_slots:
                if mat_slot.material:
                    materials.add(mat_slot.material.name)
    
    stats['material_count'] = len(materials)
    return stats


class AST_OT_InitializeProperties(Operator):
    bl_idname = "ast.initialize_properties"
    bl_label = "Initialize AST Properties"
    bl_options = {'INTERNAL'}

    ast_name: StringProperty()

    def execute(self, context):
        ast = bpy.data.objects.get(self.ast_name)
        if ast:
            ensure_ast_properties(ast)
        return {'FINISHED'}


class AST_OT_BatchExportBase(Operator):
    """Base logic for modular batch exporting (internal use)"""
    bl_options = {'INTERNAL'}

    _timer = None
    _post_status_frames = 0
    current_step = 0
    asts_to_export = []
    original_transforms = []
    original_selection = []
    error_asts = []  # Track ASTs that failed export

    def reset_state(self):
        """CRITICAL: Reset all instance variables to prevent singleton contamination"""
        self._timer = None
        self._post_status_frames = 0
        self.current_step = 0
        self.asts_to_export = []
        self.original_transforms = []
        self.original_selection = []
        self.error_asts = []

    def modal(self, context, event):
        wm = context.window_manager
        
        if event.type == 'TIMER':
            # Force panel redraw
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            
            # Check if done or error
            if wm.ast_export_error or wm.ast_export_current >= wm.ast_export_total:
                if self._post_status_frames < ASTConstants.POST_EXPORT_FRAMES:
                    self._post_status_frames += 1
                    return {'RUNNING_MODAL'}
                self.cleanup(context)
                return {'CANCELLED' if wm.ast_export_error else 'FINISHED'}

            self.process_next_export(context)
            return {'RUNNING_MODAL'}
        
        if event.type == 'ESC':
            wm.ast_export_error = True
            wm.ast_export_message = "Cancelled"
            if wm.ast_export_current < len(wm.ast_export_items):
                wm.ast_export_items[wm.ast_export_current].status = "error"
                # Track current AST as error for selection
                if wm.ast_export_current < len(self.asts_to_export):
                    self.error_asts.append(self.asts_to_export[wm.ast_export_current])
            self.cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def process_next_export(self, context):
        """Process the next AST export"""
        wm = context.window_manager
        idx = wm.ast_export_current
        
        if idx >= len(self.asts_to_export):
            return
        
        ast = self.asts_to_export[idx]
        item = wm.ast_export_items[idx]
        item.status = "active"
        
        try:
            item.stage = "Validating..."
            is_valid, error_msg = validate_ast_fields(ast)
            if not is_valid:
                raise Exception(error_msg)
            
            children = get_ast_children(ast)
            if not children:
                # No children - mark as skipped and continue
                ast["ast_active"] = False
                item.status = "error"
                item.stage = "No children - skipped"
                self.error_asts.append(ast)  # Track for selection
                self.report({'WARNING'}, f"{ast.name}: No children to export - marked inactive")
                print(f"SKIPPED: {ast.name}: No children")
                wm.ast_export_current += 1
                return
            
            item.stage = "Moving to origin..."
            original_transform = {
                'location': ast.location.copy(),
                'rotation': ast.rotation_euler.copy(),
                'scale': ast.scale.copy()
            }
            self.original_transforms.append((ast, original_transform))
            
            # Store original child scales and set to 1
            original_child_scales = []
            for child in children:
                original_child_scales.append((child, child.scale.copy()))
                child.scale = (1, 1, 1)
            
            ast.location = (0, 0, 0)
            ast.rotation_euler = (0, 0, 0)
            ast.scale = (1, 1, 1)
            bpy.context.view_layer.update()
            
            item.stage = "Selecting..."
            bpy.ops.object.select_all(action='DESELECT')
            # Don't select AST empty - only select children
            for child in children:
                if child.name in bpy.data.objects:
                    bpy.data.objects[child.name].select_set(True)
            bpy.context.view_layer.update()
            
            asset_name = ast.get("asset_name", "").strip()
            export_path = ast.get("export_path", "").strip()
            full_path = os.path.join(export_path, f"{asset_name}.fbx")
            
            item.stage = "Creating directory..."
            export_dir = os.path.dirname(full_path)
            try:
                os.makedirs(export_dir, exist_ok=True)
            except OSError as e:
                raise Exception(f"Cannot create directory: {e}")
            
            item.stage = "Exporting FBX..."
            bpy.ops.export_scene.fbx(
                filepath=full_path,
                global_scale=1.00,
                use_selection=True,
                axis_forward='-Y',
                axis_up='Z',
            )
            if not os.path.exists(full_path):
                raise Exception("FBX not created")
            
            # Restore child scales
            for child, orig_scale in original_child_scales:
                child.scale = orig_scale
            
            # Restore transform
            ast.location = original_transform['location']
            ast.rotation_euler = original_transform['rotation']
            ast.scale = original_transform['scale']
            
            item.status = "done"
            item.stage = ""
            print(f"Exported: {ast.name} -> {full_path}")
            
        except Exception as e:
            wm.ast_export_error = True
            wm.ast_export_message = str(e)
            item.status = "error"
            item.stage = ""
            self.error_asts.append(ast)  # Track for selection
            self.report({'ERROR'}, f"{ast.name}: {e}")
            print(f"ERROR: {ast.name}: {e}")
            return
        
        wm.ast_export_current += 1

    def _setup_export(self, context, active_asts, selected_objects):
        print(f"AST Export: Setting up export for {len(active_asts)} items")
        wm = context.window_manager
        wm.ast_export_active = True
        wm.ast_export_current = 0
        wm.ast_export_total = len(active_asts)
        wm.ast_export_error = False
        wm.ast_export_message = ""
        
        # Populate items list
        wm.ast_export_items.clear()
        for ast in active_asts:
            item = wm.ast_export_items.add()
            item.name = ast.name
            item.status = "pending"
            item.stage = ""
        
        # Store for processing
        self.asts_to_export = list(active_asts)
        self.current_step = 0
        self._post_status_frames = 0
        self.original_selection = list(selected_objects)
        self.original_transforms = []
        self.error_asts = []  # Reset error tracking
        
        # Add timer
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None
        
        # Restore transforms
        for ast, transform in self.original_transforms:
            ast.location = transform['location']
            ast.rotation_euler = transform['rotation']
            ast.scale = transform['scale']
        
        # Select problematic ASTs if any errors occurred, otherwise restore original selection
        bpy.ops.object.select_all(action='DESELECT')
        
        if self.error_asts:
            for ast in self.error_asts:
                if ast.name in bpy.data.objects:
                    bpy.data.objects[ast.name].select_set(True)
            if self.error_asts:
                context.view_layer.objects.active = self.error_asts[0]
        else:
            for obj in self.original_selection:
                if obj.name in bpy.data.objects:
                    bpy.data.objects[obj.name].select_set(True)
        
        wm.ast_export_active = False

    def cancel(self, context):
        self.cleanup(context)


class AST_OT_BatchExportAll(AST_OT_BatchExportBase):
    """Batch export ALL active ASTs in the scene"""
    bl_idname = "ast.batch_export_all"
    bl_label = "Export All"

    def invoke(self, context, event):
        self.reset_state()

        all_asts = get_all_asts(context)
        
        if not all_asts:
            self.report({'WARNING'}, "No ASTs found in scene")
            return {'CANCELLED'}
        
        # Filter active ASTs
        active_asts = [ast for ast in all_asts if ast.get("ast_active", True)]
        
        if not active_asts:
            self.report({'WARNING'}, "All ASTs are inactive")
            return {'CANCELLED'}
        
        return self._setup_export(context, active_asts, list(context.selected_objects))


class AST_OT_BatchExportSelected(AST_OT_BatchExportBase):
    """Batch export SELECTED active ASTs in the viewport"""
    bl_idname = "ast.batch_export_selected"
    bl_label = "Export Selected"

    def invoke(self, context, event):
        self.reset_state()

        selected_asts = get_selected_asts(context)

        if not selected_asts:
            self.report({'WARNING'}, "No ASTs selected (select an AST or its children)")
            return {'CANCELLED'}

        # Filter active ASTs from the selected set (sorted alphabetically)
        active_asts = sorted(
            [ast for ast in selected_asts if ast.get("ast_active", True)],
            key=lambda x: x.name.lower()
        )

        if not active_asts:
            self.report({'WARNING'}, f"Selected {len(selected_asts)} AST(s) are all marked as INACTIVE")
            return {'CANCELLED'}

        return self._setup_export(context, active_asts, list(context.selected_objects))


class AST_OT_SelectAllASTs(Operator):
    """Select all AST empties in the scene"""
    bl_idname = "ast.select_all"
    bl_label = "Select All"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        ast_empties = get_all_asts(context)
        
        if not ast_empties:
            self.report({'WARNING'}, "No ASTs found in scene")
            return {'CANCELLED'}
        
        bpy.ops.object.select_all(action='DESELECT')
        
        for ast in ast_empties:
            ast.select_set(True)
        
        if ast_empties:
            context.view_layer.objects.active = ast_empties[0]
        
        self.report({'INFO'}, f"Selected {len(ast_empties)} AST(s)")
        return {'FINISHED'}


class AST_OT_SelectActivated(Operator):
    """Select all ASTs that are marked as active"""
    bl_idname = "ast.select_activated"
    bl_label = "Select Activated"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        all_asts = get_all_asts(context)
        active_asts = [ast for ast in all_asts if ast.get("ast_active", True)]

        if not active_asts:
            self.report({'WARNING'}, "No active ASTs found")
            return {'CANCELLED'}
        
        bpy.ops.object.select_all(action='DESELECT')
        
        for ast in active_asts:
            ast.select_set(True)
        
        if active_asts:
            context.view_layer.objects.active = active_asts[0]
        
        self.report({'INFO'}, f"Selected {len(active_asts)} active AST(s)")
        return {'FINISHED'}


class AST_OT_SelectParent(Operator):
    """Select the parent AST of currently selected objects"""
    bl_idname = "ast.select_parent"
    bl_label = "Select Parent"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        parent_asts = get_selected_asts(context)
        
        if not parent_asts:
            self.report({'WARNING'}, "No parent ASTs found for selection")
            return {'CANCELLED'}
        
        bpy.ops.object.select_all(action='DESELECT')
        
        for ast in parent_asts:
            ast.select_set(True)
        
        if parent_asts:
            context.view_layer.objects.active = list(parent_asts)[0]
        
        self.report({'INFO'}, f"Selected {len(parent_asts)} parent AST(s)")
        return {'FINISHED'}


class AST_OT_SelectAST(Operator):
    """Select this AST empty (Shift+Click to add to selection)"""
    bl_idname = "ast.select_ast"
    bl_label = "Select AST"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    ast_name: StringProperty()

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> Set[str]:
        ast = bpy.data.objects.get(self.ast_name)
        if not ast:
            return {'CANCELLED'}
        
        is_selected = ast.select_get()
        
        if event.shift:
            ast.select_set(not is_selected)
            if not is_selected:
                context.view_layer.objects.active = ast
        else:
            if is_selected:
                ast.select_set(False)
                if context.view_layer.objects.active == ast:
                    context.view_layer.objects.active = None
            else:
                bpy.ops.object.select_all(action='DESELECT')
                ast.select_set(True)
                context.view_layer.objects.active = ast
        
        return {'FINISHED'}


class AST_OT_OpenExportPath(Operator):
    """Open export path in file explorer"""
    bl_idname = "ast.open_export_path"
    bl_label = "Open Export Path"
    bl_options = {'INTERNAL'}

    path: StringProperty()

    def execute(self, context):
        path = self.path.strip()
        if not path:
            self.report({'WARNING'}, "Export path is empty")
            return {'CANCELLED'}
        
        # Find the closest existing directory
        check_path = path
        while check_path and not os.path.exists(check_path):
            check_path = os.path.dirname(check_path)
        
        if not check_path:
            self.report({'WARNING'}, "Path does not exist")
            return {'CANCELLED'}
        
        # Open in file explorer (Windows)
        subprocess.Popen(['explorer', check_path])
        return {'FINISHED'}


class AST_OT_ToggleAllASTs(Operator):
    """Toggle all ASTs active/inactive"""
    bl_idname = "ast.toggle_all_active"
    bl_label = "Toggle All Active"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        ast_empties = get_all_asts(context)
        
        if not ast_empties:
            return {'CANCELLED'}
        
        all_active = all(ast.get("ast_active", True) for ast in ast_empties)
        new_state = not all_active
        for ast in ast_empties:
            ast["ast_active"] = new_state
        
        return {'FINISHED'}


class AST_OT_ExpandCollapseAll(Operator):
    """Expand or collapse all AST dropdowns"""
    bl_idname = "ast.expand_collapse_all"
    bl_label = "Expand/Collapse All"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        ast_empties = get_all_asts(context)
        
        if not ast_empties:
            return {'CANCELLED'}
        
        all_expanded = all(ast.get("ast_expanded", True) for ast in ast_empties)
        new_state = not all_expanded
        for ast in ast_empties:
            ast["ast_expanded"] = new_state
        
        return {'FINISHED'}


class AST_OT_ActivateDeactivateSelected(Operator):
    """Toggle active state for selected ASTs or parent ASTs of selected objects"""
    bl_idname = "ast.activate_deactivate_selected"
    bl_label = "Toggle Selected Active"
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        selected_asts = get_selected_asts(context)
        
        if not selected_asts:
            self.report({'WARNING'}, "No ASTs selected")
            return {'CANCELLED'}
        
        all_active = all(ast.get("ast_active", True) for ast in selected_asts)
        new_state = not all_active
        for ast in selected_asts:
            ast["ast_active"] = new_state
        
        return {'FINISHED'}




class AST_OT_DryRun(Operator):
    """Validate ASTs without exporting - checks all requirements"""
    bl_idname = "ast.dry_run"
    bl_label = "Dry Run"
    bl_options = {'REGISTER'}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        selected_objects = context.selected_objects
        
        if not selected_objects:
            self.report({'WARNING'}, "No objects selected")
            return {'CANCELLED'}
        
        # Find parent ASTs and identify objects without ASTs
        parent_asts = get_selected_asts(context)
        objects_without_ast = [obj for obj in selected_objects if not find_parent_ast(obj)]
        
        if objects_without_ast:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in objects_without_ast:
                obj.select_set(True)
            self.report({'ERROR'}, f"Objects without AST: {', '.join([o.name for o in objects_without_ast])}")
            return {'CANCELLED'}
        
        if not parent_asts:
            self.report({'ERROR'}, "No AST Detected")
            return {'CANCELLED'}
        
        # Filter active ASTs
        active_asts = [ast for ast in parent_asts if ast.get("ast_active", True)]
        
        if not active_asts:
            bpy.ops.object.select_all(action='DESELECT')
            for ast in parent_asts:
                ast.select_set(True)
            if parent_asts:
                context.view_layer.objects.active = list(parent_asts)[0]
            self.report({'WARNING'}, "All parent ASTs are inactive")
            return {'CANCELLED'}
        
        errors = []
        valid_count = 0
        
        for ast in sorted(active_asts, key=lambda x: x.name.lower()):
            ast_errors = []
            
            is_valid, error_msg = validate_ast_fields(ast)
            if not is_valid:
                ast_errors.append(error_msg)
            
            children = get_ast_children(ast)
            if not children:
                ast_errors.append("No children")
            
            export_path = ast.get("export_path", "").strip()
            if export_path:
                try:
                    if not os.path.exists(export_path):
                        parent_dir = os.path.dirname(export_path)
                        if parent_dir and not os.path.exists(parent_dir):
                            ast_errors.append(f"Path does not exist: {parent_dir}")
                except Exception as e:
                    ast_errors.append(f"Path error: {e}")

            if ast_errors:
                errors.append(f"{ast.name}: {'; '.join(ast_errors)}")
            else:
                valid_count += 1
        
        total = len(active_asts)
        if errors:
            bpy.ops.object.select_all(action='DESELECT')
            error_ast_names = [error.split(':')[0] for error in errors]
            for ast in active_asts:
                if ast.name in error_ast_names:
                    ast.select_set(True)
            if error_ast_names:
                first_error_ast = next((ast for ast in active_asts if ast.name in error_ast_names), None)
                if first_error_ast:
                    context.view_layer.objects.active = first_error_ast
            
            for error in errors:
                self.report({'ERROR'}, error)
            self.report({'WARNING'}, f"Dry Run: {valid_count}/{total} ASTs valid")
            return {'CANCELLED'}
        else:
            self.report({'INFO'}, f"Dry Run: All {total} AST(s) ready to export")
            return {'FINISHED'}


class AST_OT_OrganiseASTs(Operator):
    """Arrange selected ASTs on a grid"""
    bl_idname = "ast.organise_asts"
    bl_label = "Organise ASTs"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        context.view_layer.update()
        wm = context.window_manager
        scope = wm.ast_organise_scope
        orientation = wm.ast_organise_orientation
        use_bbox = wm.ast_organise_bbox
        
        if scope == 'SELECTED':
            objs = list(get_selected_asts(context))
        else:
            objs = get_all_asts(context)

        if not objs:
            self.report({'WARNING'}, f"No ASTs found ({scope.lower()})")
            return {'CANCELLED'}
            
        objs.sort(key=lambda x: x.name.lower())
        
        master_offset = ASTConstants.GRID_SPACING # Start grid spacing
        padding = 1.0
        
        for o in objs:
            if use_bbox:
                wmin_x, wmax_x, wmin_y, wmax_y = get_ast_world_bounds(o)
                origin = o.matrix_world.translation
                
                # Snap the current pointer to the next grid mark BEFORE placing
                current_mark = math.ceil(master_offset / ASTConstants.GRID_SPACING) * ASTConstants.GRID_SPACING
                
                if orientation == 'HORIZONTAL':
                    lmin = wmin_y - origin.y
                    lmax = wmax_y - origin.y
                    pos_val = float(math.ceil((current_mark - lmin) / ASTConstants.GRID_SPACING) * ASTConstants.GRID_SPACING)
                    o.matrix_world.translation = Vector((0.0, pos_val, 0.0))
                    master_offset = pos_val + lmax + padding
                else:
                    lmin = wmin_x - origin.x
                    lmax = wmax_x - origin.x
                    pos_val = float(math.ceil((current_mark - lmin) / ASTConstants.GRID_SPACING) * ASTConstants.GRID_SPACING)
                    o.matrix_world.translation = Vector((pos_val, 0.0, 0.0))
                    master_offset = pos_val + lmax + padding
            else:
                if orientation == 'HORIZONTAL':
                    o.matrix_world.translation = Vector((0.0, master_offset, 0.0))
                else:
                    o.matrix_world.translation = Vector((master_offset, 0.0, 0.0))
                master_offset += ASTConstants.GRID_SPACING
            
        self.report({'INFO'}, f"Organised {len(objs)} AST(s) to grid. Mode: {'Dynamic' if use_bbox else 'Fixed'} {ASTConstants.GRID_SPACING}m")
        return {'FINISHED'}


def get_ast_world_bounds(ast_obj: bpy.types.Object) -> Tuple[float, float, float, float]:
    """
    Calculate world-space boundaries of AST.
    
    Args:
        ast_obj: The AST empty object.
        
    Returns:
        Tuple of (min_x, max_x, min_y, max_y).
    """
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    
    objects_to_check = [ast_obj] + [child for child in ast_obj.children_recursive]
    
    found_geometry = False
    for obj in objects_to_check:
        if obj.type == 'MESH' and obj.bound_box:
            found_geometry = True
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                min_x = min(min_x, world_corner.x)
                min_y = min(min_y, world_corner.y)
                min_z = min(min_z, world_corner.z)
                max_x = max(max_x, world_corner.x)
                max_y = max(max_y, world_corner.y)
                max_z = max(max_z, world_corner.z)
                
    if not found_geometry:
        # Fallback to current location +- 1m
        loc = ast_obj.matrix_world.translation
        return loc.x - 1.0, loc.x + 1.0, loc.y - 1.0, loc.y + 1.0
        
    return min_x, max_x, min_y, max_y





class AST_OT_OrganiseByGroup(Operator):
    """Arrange all ASTs in the scene based on their group assignment"""
    bl_idname = "ast.organise_by_group"
    bl_label = "Organise by Group"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context: bpy.types.Context) -> Set[str]:
        context.view_layer.update()
        wm = context.window_manager
        orientation = wm.ast_organise_orientation
        use_bbox = wm.ast_organise_bbox
        scope = wm.ast_organise_scope

        if scope == 'SELECTED':
            objs = list(get_selected_asts(context))
        else:
            objs = get_all_asts(context)

        if not objs:
            return {'CANCELLED'}

        # Groups mapping
        unique_groups = sorted(list(set(o.ast_group_index for o in objs)))
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        
        # Rank mapping (sort by group then name)
        objs.sort(key=lambda x: (x.ast_group_index, x.name.lower()))
        rank_to_objs: Dict[int, List[bpy.types.Object]] = {}
        obj_to_rank: Dict[bpy.types.Object, int] = {}
        group_counters = {g: 0 for g in unique_groups}
        
        for o in objs:
            g = o.ast_group_index
            rank = group_counters[g]
            obj_to_rank[o] = rank
            rank_to_objs.setdefault(rank, []).append(o)
            group_counters[g] += 1

        padding = 1.0
        lane_offsets: Dict[int, float] = {}
        rank_offsets: Dict[int, float] = {}

        if use_bbox:
            # DYNAMIC 2D GRID
            curr_lane_frontier = 0.0
            for idx in range(len(unique_groups)):
                g_id = unique_groups[idx]
                lane_asts = [o for o in objs if o.ast_group_index == g_id]
                
                best_l = curr_lane_frontier
                for o in lane_asts:
                    wmin_x, wmax_x, wmin_y, wmax_y = get_ast_world_bounds(o)
                    origin = o.matrix_world.translation
                    lmin = (wmin_x - origin.x) if orientation == 'HORIZONTAL' else (wmin_y - origin.y)
                    best_l = max(best_l, curr_lane_frontier - lmin)
                
                best_l = math.ceil(best_l / ASTConstants.GRID_SPACING) * ASTConstants.GRID_SPACING
                lane_offsets[idx] = best_l
                
                lane_f = best_l
                for o in lane_asts:
                    wmin_x, wmax_x, wmin_y, wmax_y = get_ast_world_bounds(o)
                    origin = o.matrix_world.translation
                    lmax = (wmax_x - origin.x) if orientation == 'HORIZONTAL' else (wmax_y - origin.y)
                    lane_f = max(lane_f, best_l + lmax + padding)
                curr_lane_frontier = lane_f

            curr_rank_frontier = 0.0
            sorted_ranks = sorted(rank_to_objs.keys())
            for r in sorted_ranks:
                r_asts = rank_to_objs[r]
                best_r = curr_rank_frontier
                for o in r_asts:
                    wmin_x, wmax_x, wmin_y, wmax_y = get_ast_world_bounds(o)
                    origin = o.matrix_world.translation
                    lmin = (wmin_y - origin.y) if orientation == 'HORIZONTAL' else (wmin_x - origin.x)
                    best_r = max(best_r, curr_rank_frontier - lmin)
                
                best_r = math.ceil(best_r / ASTConstants.GRID_SPACING) * ASTConstants.GRID_SPACING
                rank_offsets[r] = best_r
                
                rank_f = best_r
                for o in r_asts:
                    wmin_x, wmax_x, wmin_y, wmax_y = get_ast_world_bounds(o)
                    origin = o.matrix_world.translation
                    lmax = (wmax_y - origin.y) if orientation == 'HORIZONTAL' else (wmax_x - origin.x)
                    rank_f = max(rank_f, best_r + lmax + padding)
                curr_rank_frontier = rank_f
        else:
            # FIXED GRID
            for idx in range(len(unique_groups)):
                lane_offsets[idx] = idx * ASTConstants.GRID_SPACING
            for r in rank_to_objs:
                rank_offsets[r] = r * ASTConstants.GRID_SPACING

        # Apply positions
        for o in objs:
            g = o.ast_group_index
            idx = group_to_idx[g]
            r = obj_to_rank[o]
            
            l_coord = lane_offsets[idx]
            f_coord = rank_offsets[r]
            
            if orientation == 'HORIZONTAL':
                o.matrix_world.translation = Vector((l_coord, f_coord, 0.0))
            else:
                o.matrix_world.translation = Vector((f_coord, l_coord, 0.0))

        self.report({'INFO'}, f"Organised by Group ({'Dynamic' if use_bbox else 'Fixed'} {ASTConstants.GRID_SPACING}m).")
        return {'FINISHED'}


class AST_OT_ShowGroupOverlay(Operator):
    """Show group numbers for all ASTs in the scene"""
    bl_idname = "ast.show_group_overlay"
    bl_label = "Show Group Numbers"
    
    _handle = None
    _first_release = False
    
    def draw_callback_px(self):
        font_id = 0
        context = bpy.context
        region = context.region
        rv3d = context.space_data.region_3d
        
        if not region or not rv3d:
            return

        for obj in context.scene.objects:
            if obj.type == 'EMPTY' and obj.name.startswith('AST_'):
                group = obj.ast_group_index
                if group != -1:

                    pos = obj.matrix_world.translation
                    # Project 3D point to 2D region
                    coord = view3d_utils.location_3d_to_region_2d(region, rv3d, pos)
                    
                    if coord:
                        # Draw text
                        blf.size(font_id, 24)
                        blf.color(font_id, 1, 0.8, 0, 1) # Gold
                        blf.position(font_id, coord.x + 10, coord.y + 10, 0)
                        blf.draw(font_id, str(group + 1))

    def modal(self, context, event):
        if context.area:
            context.area.tag_redraw()

        if not self._first_release:
            if event.value == 'RELEASE':
                self._first_release = True
            return {'PASS_THROUGH'}

        if event.type in {'LEFTMOUSE', 'RIGHTMOUSE', 'ESC', 'RET', 'SPACE'} or \
           (event.type in {'WHEELUP', 'WHEELDOWN', 'MIDDLEMOUSE'}):
             bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
             context.area.tag_redraw()
             return {'FINISHED'}
            
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':
            self._first_release = False
            self._handle = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback_px, (), 'WINDOW', 'POST_PIXEL')
            context.window_manager.modal_handler_add(self)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        return {'CANCELLED'}


class AST_OT_GroupAction(Operator):
    """Assign to Group (Click) / Select Group (Ctrl+Click)"""
    bl_idname = "ast.group_action"
    bl_label = "Group Action"
    bl_options = {'REGISTER', 'UNDO'}
    
    group_index: IntProperty()
    
    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> Set[str]:
        if event.ctrl:
            return self.select_group(context)
        else:
            return self.assign_group(context)
            
    def select_group(self, context: bpy.types.Context) -> Set[str]:
        bpy.ops.object.select_all(action='DESELECT')
        
        found = False
        all_asts = get_all_asts(context)
        for obj in all_asts:
            if obj.ast_group_index == self.group_index:
                obj.select_set(True)
                found = True
        
        if found:
            self.report({'INFO'}, f"Selected group {self.group_index + 1}")
        else:
            self.report({'WARNING'}, f"No ASTs in group {self.group_index + 1}")
            
        return {'FINISHED'}

    def assign_group(self, context: bpy.types.Context) -> Set[str]:
        selected_asts = get_selected_asts(context)
        
        if not selected_asts:
            self.report({'WARNING'}, "No ASTs selected")
            return {'CANCELLED'}
            
        count = 0
        for ast in selected_asts:
            current_group = ast.ast_group_index
            if current_group == self.group_index:
                ast.ast_group_index = -1
                self.report({'INFO'}, f"Removed {ast.name} from group {self.group_index + 1}")
            else:
                ast.ast_group_index = self.group_index
                count += 1
                
        if count > 0:
            self.report({'INFO'}, f"Assigned {count} ASTs to Group {self.group_index + 1}")
            
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
                
        return {'FINISHED'}


class AST_PT_OrganisationPanel(Panel):
    bl_label = "Studio Batch Export - Groups"
    bl_idname = "AST_PT_organisation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Export'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 20

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        
        # Organise section moved to top
        box = layout.box()
        
        # Row 1: Orientation and Scope (Split Half)
        row = box.row(align=True)
        row.prop(wm, "ast_organise_orientation", text="")
        row.prop(wm, "ast_organise_scope", text="")
        
        # Row 2: To Grid | By Group | BBox | Overlay
        row = box.row(align=True)
        row.operator("ast.organise_asts", icon='GRID', text="To Grid")
        row.operator("ast.organise_by_group", icon='GRID', text="By Group")
        row.prop(wm, "ast_organise_bbox", text="", icon='MESH_CUBE', toggle=True)
        row.operator("ast.show_group_overlay", text="", icon='RESTRICT_VIEW_OFF')
        
        layout.separator()
        
        # HIGHLIGHT LOGIC: Highlight if ANY selected AST is in the group
        active_groups = get_selected_ast_groups(context)

        # Draw 3x4 Grid manually
        grid_col = layout.column(align=True)
        for row_idx in range(3):
            row = grid_col.row(align=True)
            row.scale_y = 1.5
            for col_idx in range(4):
                i = row_idx * 4 + col_idx
                row.operator("ast.group_action", text=str(i+1), depress=(i in active_groups)).group_index = i
        
        layout.label(text="Ctrl+Click to Select All ASTs in Group", icon='INFO')


class AST_PT_BatchExportPanel(Panel):
    bl_label = "Studio Batch Export - Tools"
    bl_idname = "AST_PT_batch_export"
    bl_order = 10
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Export'

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        wm = context.window_manager

        # Get all AST empties
        ast_empties = get_all_asts(context)

        if not ast_empties:
            layout.label(text="No ASTs found in scene", icon='INFO')
            layout.label(text="Create an empty named AST_*")
            return
        
        # Check for duplicate asset names
        names = [ast.get("asset_name", "").strip() for ast in ast_empties]
        duplicate_names = {name for name in names if name and names.count(name) > 1}
        
        # Export button with Dry Run button
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.enabled = not wm.ast_export_active
        
        # Determine internal duplicates for selected ASTs
        selected_asts = get_selected_asts(context)
        
        sel_names = [ast.get("asset_name", "").strip() for ast in selected_asts]
        sel_duplicates = {name for name in sel_names if name and sel_names.count(name) > 1}
        
        # Export All
        sub_all = row.row(align=True)
        sub_all.enabled = len(duplicate_names) == 0
        sub_all.operator("ast.batch_export_all", text="All", 
                         icon='ERROR' if len(duplicate_names) > 0 else 'EXPORT')

        # Export Selected
        sub_sel = row.row(align=True)
        sub_sel.enabled = len(sel_duplicates) == 0 and len(selected_asts) > 0
        sub_sel.operator("ast.batch_export_selected", text="Selected", icon='EXPORT')
        
        # Dry Run
        row.operator("ast.dry_run", text="", icon='FILTER')
        
        # Show duplicate warning at top
        if duplicate_names:
            box = layout.box()
            box.alert = True
            box.label(text="Duplicate Asset Names!", icon='ERROR')
            for name in sorted(duplicate_names):
                box.label(text=f'"{name}"', icon='ERROR')
        
        layout.separator()
        
        # Found ASTs row
        layout.label(text=f"Found {len(ast_empties)} AST(s):")
        
        # Determine states for button icons
        initialized_asts = [ast for ast in ast_empties if "ast_active" in ast]
        all_active = (
            bool(initialized_asts) and
            all(ast.get("ast_active", True) for ast in initialized_asts)
        )
        
        expanded_asts = [ast for ast in ast_empties if "ast_expanded" in ast]
        all_expanded = (
            bool(expanded_asts) and
            all(ast.get("ast_expanded", True) for ast in expanded_asts)
        )
        
        # Check if any selected ASTs are active
        selected_asts = get_selected_asts(context)
        selected_all_active = all(ast.get("ast_active", True) for ast in selected_asts) if selected_asts else False
        
        # Row 1: Select Parent | Select Activated | Select All
        row = layout.row(align=True)
        row.operator("ast.select_parent", text="Select Parent", icon='FILE_PARENT')
        row.operator("ast.select_activated", text="Select Activated", icon='RADIOBUT_ON')
        row.operator("ast.select_all", text="Select All", icon='RESTRICT_SELECT_OFF')
        
        # Row 2: Activate Selected | Activate All
        row = layout.row(align=True)
        row.operator("ast.activate_deactivate_selected", text="Activate Selected",
                     icon='CHECKBOX_HLT' if selected_all_active else 'CHECKBOX_DEHLT')
        row.operator("ast.toggle_all_active", text="Activate All",
                     icon='CHECKBOX_HLT' if all_active else 'CHECKBOX_DEHLT')
        
        # Expand/Collapse toggle on left
        row = layout.row()
        sub = row.row(align=True)
        sub.alignment = 'LEFT'
        sub.operator("ast.expand_collapse_all", text="", emboss=False,
                     icon='TRIA_DOWN' if all_expanded else 'TRIA_RIGHT')
        sub.label(text="Expand All" if not all_expanded else "Collapse All")
        
        for ast in ast_empties:
            has_props = "asset_name" in ast and "export_path" in ast and "ast_active" in ast
            if not has_props:
                box = layout.box()
                box.label(text=f"{ast.name} (Not initialized)", icon='ERROR')
                op = box.operator("ast.initialize_properties", text="Initialize")
                op.ast_name = ast.name
                continue
            is_expanded = ast.get("ast_expanded", True)
            has_expanded_prop = "ast_expanded" in ast
            box = layout.box()
            header = box.row()
            if has_expanded_prop:
                op_row = header.row(align=True)
                op_row.prop(ast, '["ast_expanded"]', text="", emboss=False,
                         icon='TRIA_DOWN' if is_expanded else 'TRIA_RIGHT')
            else:
                op = header.operator("ast.initialize_properties", text="", icon='TRIA_DOWN', emboss=False)
                op.ast_name = ast.name
            header.label(text=ast.name)
            ast_children = get_ast_children(ast)
            is_selected = ast.select_get() or any(child.select_get() for child in ast_children)
            op = header.operator("ast.select_ast", text="", emboss=False,
                                 icon='RADIOBUT_ON' if is_selected else 'RADIOBUT_OFF')
            op.ast_name = ast.name
            header.prop(ast, '["ast_active"]', text="",
                     icon='CHECKBOX_HLT' if ast.get("ast_active", True) else 'CHECKBOX_DEHLT')
            if is_expanded:
                box.prop(ast, '["asset_name"]', text="Asset Name")
                row = box.row(align=True)
                row.prop(ast, '["export_path"]', text="Export Path")
                op = row.operator("ast.open_export_path", text="", icon='FILE_FOLDER')
                op.path = ast.get("export_path", "")
                ast_asset_name = ast.get("asset_name", "").strip()
                if ast_asset_name in duplicate_names:
                    row = box.row()
                    row.alert = True
                    row.label(text="⚠ Duplicate asset name", icon='ERROR')
                else:
                    children = get_ast_children(ast)
                    if not children:
                        row = box.row()
                        row.alert = True
                        row.label(text="⚠ No children - inactive", icon='ERROR')
                    else:
                        is_valid, error_msg = validate_ast_fields(ast)
                        if not is_valid:
                            row = box.row()
                            row.alert = True
                            row.label(text=f"⚠ {error_msg}", icon='ERROR')
                        else:
                            stats = get_ast_stats(children)
                            box.label(text=f"Children: {stats['child_count']} object(s)", icon='OUTLINER_OB_MESH')
                            box.label(text=f"Triangles: {stats['tri_count']:,}", icon='MESH_DATA')
                            box.label(text=f"Materials: {stats['material_count']}", icon='MATERIAL')
                            if stats['has_collision']:
                                box.label(text=f"Collision: {stats['collision_count']} object(s)", icon='PHYSICS')
                                box.label(text=f"Collision Tris: {stats['collision_tri_count']:,}", icon='MOD_PHYSICS')


        # === EXPORT STATUS SECTION ===
        if wm.ast_export_active or len(wm.ast_export_items) > 0:
            layout.separator()
            status_box = layout.box()
            
            # Header row with expand toggle
            header = status_box.row()
            header.prop(wm, "ast_export_status_expanded", text="", emboss=False,
                     icon='TRIA_DOWN' if wm.ast_export_status_expanded else 'TRIA_RIGHT')
            header.label(text="Export Status")
            
            if not wm.ast_export_status_expanded:
                # Show compact summary when collapsed
                done_count = sum(1 for item in wm.ast_export_items if item.status == "done")
                error_count = sum(1 for item in wm.ast_export_items if item.status == "error")
                total = len(wm.ast_export_items)

                # Create a sub-row aligned to the left
                sub = header.row()
                sub.alignment = 'LEFT'
                if error_count > 0:
                    sub.label(text=f"{done_count}/{total} (errors: {error_count})", icon='ERROR')
                elif wm.ast_export_active:
                    sub.label(text=f"{done_count}/{total}", icon='TIME')
                else:
                    sub.label(text=f"{done_count}/{total}", icon='CHECKMARK')
            else:
                # Item list with status
                for item in wm.ast_export_items:
                    row = status_box.row()
                    if item.status == "done":
                        row.label(text=item.name, icon='CHECKMARK')
                    elif item.status == "active":
                        row.label(text=item.name, icon='PLAY')
                        if item.stage:
                            sub = status_box.row()
                            sub.label(text=item.stage, icon='BLANK1')
                    elif item.status == "error":
                        row.alert = True
                        row.label(text=item.name, icon='ERROR')
                    else:  # pending
                        row.label(text=item.name, icon='LAYER_USED')
                
                # Error message
                if wm.ast_export_error and wm.ast_export_message:
                    row = status_box.row()
                    row.alert = True
                    row.label(text=f"ERROR: {wm.ast_export_message}")

def register():
    bpy.types.Object.ast_group_index = IntProperty(
        name="AST Group Index",
        default=-1,
        min=-1,
        max=11,
        description="Organization group index (0-11)"
    )
    
    bpy.utils.register_class(AST_ExportItemPG)
    bpy.utils.register_class(AST_OT_InitializeProperties)
    bpy.utils.register_class(AST_OT_GroupAction)
    bpy.utils.register_class(AST_PT_OrganisationPanel)
    bpy.utils.register_class(AST_OT_BatchExportAll)
    bpy.utils.register_class(AST_OT_BatchExportSelected)
    bpy.utils.register_class(AST_OT_SelectAllASTs)
    bpy.utils.register_class(AST_OT_SelectActivated)
    bpy.utils.register_class(AST_OT_SelectParent)
    bpy.utils.register_class(AST_OT_SelectAST)
    bpy.utils.register_class(AST_OT_OpenExportPath)
    bpy.utils.register_class(AST_OT_ToggleAllASTs)
    bpy.utils.register_class(AST_OT_ExpandCollapseAll)
    bpy.utils.register_class(AST_OT_ActivateDeactivateSelected)
    bpy.utils.register_class(AST_OT_DryRun)
    bpy.utils.register_class(AST_OT_OrganiseByGroup)
    bpy.utils.register_class(AST_OT_ShowGroupOverlay)
    bpy.utils.register_class(AST_OT_OrganiseASTs)
    bpy.utils.register_class(AST_PT_BatchExportPanel)
    
    # WindowManager properties for export state
    bpy.types.WindowManager.ast_export_active = BoolProperty(default=False)
    bpy.types.WindowManager.ast_export_current = IntProperty(default=0)
    bpy.types.WindowManager.ast_export_total = IntProperty(default=0)
    bpy.types.WindowManager.ast_export_error = BoolProperty(default=False)
    bpy.types.WindowManager.ast_export_message = StringProperty(default="")
    bpy.types.WindowManager.ast_export_items = CollectionProperty(type=AST_ExportItemPG)
    bpy.types.WindowManager.ast_export_status_expanded = BoolProperty(
        name="Expanded Status",
        default=False
    )
    
    # New properties for Organisation Panel
    bpy.types.WindowManager.ast_organise_orientation = EnumProperty(
        name="Orientation",
        items=[
            ('HORIZONTAL', "Horizontal", "Group columns along X, Items along Y"),
            ('VERTICAL', "Vertical", "Group rows along Y, Items along X"),
        ],
        default='HORIZONTAL'
    )
    bpy.types.WindowManager.ast_organise_bbox = BoolProperty(
        name="Enable Bounding Box",
        description="Use cumulative bounding box size for spacing",
        default=False
    )
    bpy.types.WindowManager.ast_organise_scope = EnumProperty(
        name="Organise Scope",
        items=[
            ('ALL', "All ASTs", "Organise all ASTs in the scene"),
            ('SELECTED', "Selected Only", "Organise only selected ASTs"),
        ],
        default='ALL'
    )


def unregister():
    # Safe cleanup of WindowManager properties
    for prop in [
        "ast_export_active", "ast_export_current", "ast_export_total",
        "ast_export_error", "ast_export_message", "ast_export_items",
        "ast_export_status_expanded", "ast_organise_orientation",
        "ast_organise_bbox", "ast_organise_scope"
    ]:
        if hasattr(bpy.types.WindowManager, prop):
            delattr(bpy.types.WindowManager, prop)
    
    bpy.utils.unregister_class(AST_PT_BatchExportPanel)
    bpy.utils.unregister_class(AST_OT_OrganiseASTs)
    bpy.utils.unregister_class(AST_OT_ShowGroupOverlay)
    bpy.utils.unregister_class(AST_OT_OrganiseByGroup)
    bpy.utils.unregister_class(AST_OT_DryRun)
    bpy.utils.unregister_class(AST_OT_ActivateDeactivateSelected)
    bpy.utils.unregister_class(AST_OT_ExpandCollapseAll)
    bpy.utils.unregister_class(AST_OT_ToggleAllASTs)
    bpy.utils.unregister_class(AST_OT_OpenExportPath)
    bpy.utils.unregister_class(AST_OT_SelectAST)
    bpy.utils.unregister_class(AST_OT_SelectParent)
    bpy.utils.unregister_class(AST_OT_SelectActivated)
    bpy.utils.unregister_class(AST_OT_SelectAllASTs)
    bpy.utils.unregister_class(AST_OT_BatchExportSelected)
    bpy.utils.unregister_class(AST_OT_BatchExportAll)
    bpy.utils.unregister_class(AST_PT_OrganisationPanel)
    bpy.utils.unregister_class(AST_OT_GroupAction)
    bpy.utils.unregister_class(AST_OT_InitializeProperties)
    bpy.utils.unregister_class(AST_ExportItemPG)
    
    del bpy.types.Object.ast_group_index


if __name__ == "__main__":
    register()
