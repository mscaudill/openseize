::: openseize.filtering.iir.Cheby2
    options:    
        members_order:
            source  
        show_source: 
            true
        show_root_heading:
            true
        show_root_toc_entry: 
            false

## **Bases and Mixins**

## IIR Base
::: openseize.filtering.bases.IIR
    options:    
        members_order:
            source
        members:
            - btype
            - ftype
            - pass_attenuation
            - cutoff
            - __call__
        show_source: 
            false
        show_root_heading:
            false
        show_root_toc_entry: 
            false

## Viewer Mixin
::: openseize.filtering.mixins.Viewer
    options:
        show_source: 
            false
        members:
            - plot
        show_root_heading:
            false
        show_root_toc_entry: 
            false

