"""Two nested non-rigid frames and a camera, composed before touching vertices."""

from numga import NumpyContext
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Line = ga.gatype.bivector()


# --- math -----------------------------------------------------------------------------
def main() -> Point:
    # A triangle in the child frame; homogeneous coordinates (x, y, z, weight).
    vertices = mv.antivector([[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0, 0.5, 0, 1]])
    parent_pose = (mv.zw * 2.5).exp() * (mv.xy * 0.3).exp()
    child_pose = (mv.xw * 0.5).exp() * (mv.xy * -0.4 + mv.yz * 0.2).exp()
    camera_pose = (mv.xw * 0.25).exp() * (mv.yz * 0.06).exp()
    lens_center = mv.zyx
    lens_plane = -mv.z                      # Light travels towards negative camera z.
    focal_length = 1.0
    pupil = (mv.xw * 0.05).exp() >> lens_center
    sensor_plane = mv.z + 1.25 * mv.w        # z = -1.25, focused at object distance 5.

    # Scale along local axes, then rotate and translate into the enclosing frame.
    parent_scale = Point + mv.yzw * (mv.x & Point)         # x *= 2
    child_scale = Point - 0.5 * mv.zxw * (mv.y & Point)   # y *= 0.5
    parent_to_world = parent_pose >> parent_scale
    child_to_parent = child_pose >> child_scale

    # Pick one off-centre pupil point: rays through the lens centre would not bend.
    world_to_camera = camera_pose << Point
    incoming_rays = world_to_camera & pupil
    lens = Line - (lens_center & (Line ^ lens_plane)) / focal_length
    to_sensor = Line ^ sensor_plane
    camera = to_sensor(lens(incoming_rays))

    # 640 x 480 pixels, 400 pixels per sensor unit, image y increasing downwards.
    viewport = (mv.yzw * (400 * (mv.x & Point) + 320 * (mv.w & Point))
                + mv.zxw * (-400 * (mv.y & Point) + 240 * (mv.w & Point))
                + mv.zyx * (mv.w & Point))

    # One extensor for the entire chain; one application to the vertex batch, 16 fmadds per vertex.
    local_to_pixel = viewport(camera(parent_to_world(child_to_parent)))
    pixels = local_to_pixel(vertices)
    return pixels / (mv.w & pixels)  # Perspective division is the final readout.


# --- plumbing -------------------------------------------------------------------------
if __name__ == "__main__":
    pixels = main()
    print("pixel x:", (mv.x & pixels).to_array())
    print("pixel y:", (mv.y & pixels).to_array())
