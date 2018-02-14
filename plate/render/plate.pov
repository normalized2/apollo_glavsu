// This work is licensed under the Creative Commons Attribution 3.0 Unported License.
// To view a copy of this license, visit http://creativecommons.org/licenses/by/3.0/
// or send a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
// California, 94041, USA.

// Persistence Of Vision raytracer sample file.
// File: phot_met_glass.pov
// Desc: metal, glass and photons sample
// Date: August 2001
// Auth: Christoph Hormann

// -w320 -h160
// -w512 -h256 +a0.3

#version 3.6;

#include "colors.inc"
#include "glass.inc"

global_settings {
  assumed_gamma 1.0
  max_trace_level 25
  photons {
    spacing 0.03
    autostop 0
    jitter 0
  }
  ambient_light rgb <0.1, 0.1, 0.1>
}

camera {
  location    <0, 10, 1>
  right       2*x
  look_at     <0,-1,0>
  angle       34
}

// light_source {
//   <-0, 18, 0>
//   color rgb <1.3, 1.2, 1.1>

//  photons {
//    reflection on
//    refraction on
//  }
//}


#declare light1 =
light_source {
   <0, 10, -0> color red 1.0 green 0.1 blue 0.1
   spotlight
   point_at <0, 1, 0>
   tightness 20
   radius 1
   falloff 1
}


#declare light2 =
light_source {
   <0, 20, -0> color red 30.0 green 0.0 blue 0.0
   cylinder
   point_at <0, 1, 0>
   tightness 0
   radius 0.02
   falloff 0.02
}

light_source { light2 }


plane {
  y, -1
  texture {
    pigment {
      color rgb 1.0
    }
  }
}


#declare Metal_Texture =
texture {
  pigment { color rgb <0.5, 0.5, 0.6> }
  finish {
    ambient 0.0
    diffuse 0.15
    specular 0.3
    roughness 0.01
    reflection {
      0.8
      metallic
    }
  }
}


#declare glass_material_1 =
    material {
      texture {
        pigment { color Col_Glass_Clear }
        finish {
          specular 0.6
          roughness 0.002
          ambient 0
          diffuse 0.1
          brilliance 5
          reflection {
            0.1, 1.0
            fresnel on
          }
          conserve_energy
        }
      }
      interior {
        ior 1.5
        //I_Glass_Exp(2)
        //fade_color Col_Red_03
      }
    }



// Create plate
#declare plate = box {
  <-5, -1, -5>  // one corner position <X1 Y1 Z1>
  <5, -0.6, 5>  // other corner position <X2 Y2 Z2>

  // texture { Metal_Texture }
  material { glass_material_1 }
}


// declare glass ball
#declare ball = sphere {
    <0,0,0>, 1

    material { glass_material_1 }
  }



union {
  //object { Metal_Obj_2 }

  object { plate }

  //object { ball  translate <0,2,0> }

  photons {
    target
    reflection on
    refraction on
    //collect off
  }
}

