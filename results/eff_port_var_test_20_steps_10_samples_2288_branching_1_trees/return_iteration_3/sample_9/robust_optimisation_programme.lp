\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.23743 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.922428 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.06721 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.832551 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.02072 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.1571 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.16844 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.904825 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.46351 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.878709 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.12563 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.747345 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.976022 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.897242 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.19452 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.03071 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.27463 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.23908 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.990042 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.720329 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.08902 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.874275 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.09131 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.994973 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.42695 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.797212 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.14217 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.52733 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.893967 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.859171 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.0363 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.201 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.784067 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.20289 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.26127 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.06416 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.904492 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.27777 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.28146 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.89648 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.22109 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.794786 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.01588 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.07569 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.11425 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.700272 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.78775 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.807883 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.29382 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.670742 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.13069 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.079 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.18395 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.08907 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.3795 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.62531 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.15948 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.665754 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.902887 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.835414 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.01 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.09284 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.67518 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.880406 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.24282 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.560423 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.860958 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.695144 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.852317 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.49806 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.695167 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.35994 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.64242 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.19369 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.07693 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.60797 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.33828 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.863951 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.26445 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.778662 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.10446 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.10854 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.10935 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.778625 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.96953 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.557654 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.61128 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.787289 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.27181 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.945439 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.16526 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.08502 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.69094 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.30174 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.17283 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.646513 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.891938 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.823961 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.19786 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.00672 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.78957 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.882092 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.34568 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.424856 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.833469 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.577637 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.853242 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.37067 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.610217 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.59957 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.36873 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.11922 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.905166 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.54226 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19408 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.911339 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.20654 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.866352 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.965642 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.12587 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.18012 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.953791 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.45123 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.769778 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.34242 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.776524 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.960877 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.947667 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.27973 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.992976 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.36568 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.22677 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.957502 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.753072 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.975596 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.78504 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.17082 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.10042 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.46735 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.858432 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.3005 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.557107 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.916895 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.818893 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.97402 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.17453 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.705418 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.32642 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.21871 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.16378 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.917295 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.40731 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15383 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.945155 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.21965 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.924576 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.16701 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.830505 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.731452 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.501971 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.59748 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.671845 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.0532 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.940194 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.73962 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.07577 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.586073 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.05637 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.27062 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.28275 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.69901 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.679006 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.729118 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.01749 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.719284 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.994366 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.61516 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.28032 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.2057 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.706591 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.80861 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.628202 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.665875 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.77605 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.727962 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.23812 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.32288 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.787608 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.17678 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.02479 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.12297 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.947135 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.15517 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.915899 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.00312 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.07096 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.11925 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.996304 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.41825 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.842618 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.23037 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.886833 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.936399 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.872566 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.0554 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.91263 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.2667 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.12501 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.06745 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.71805 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.04546 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.953999 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.12066 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.07744 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.33218 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.921766 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.3953 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.683094 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.847726 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.763585 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.931043 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.00778 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.769105 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.18511 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.24 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.09396 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.892926 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.24634 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1641 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.943775 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.21619 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.89871 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.0298 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.87221 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.965817 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.658403 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.59059 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.656534 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.20433 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.880621 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.33372 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.982177 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.890284 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.04472 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.39057 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.40349 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.42499 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.689333 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.883524 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.913325 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.990983 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.934357 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.62688 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.23082 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.27274 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.673667 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.984858 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.717113 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.878597 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.34717 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.814922 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.34838 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.38333 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.08677 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.02028 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.32604 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.08156 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.903412 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.37908 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.983425 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.13734 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.855499 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.948497 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.56297 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.94968 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.66128 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.14154 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.747932 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.7553 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.40345 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.657969 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.04963 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.990333 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.60103 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.75461 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.627636 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.806397 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.903687 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.629942 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.05231 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 2.02727 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.27852 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.16303 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.752361 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.892913 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.616057 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.90938 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 2.58996 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.746096 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.0646 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.65825 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.886119 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.30529 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.990607 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14127 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.905044 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.16513 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.961794 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.04942 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.06229 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.15452 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.971188 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.61132 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.748449 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.37864 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.824037 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.860246 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.913866 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.05972 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.0116 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.19793 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.02501 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.978349 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.666805 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.901321 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.851655 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.09427 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.03132 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.37429 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.921412 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.21097 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.617379 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.905045 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.800046 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.07258 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.16586 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.76371 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.32257 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.27322 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.1424 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.943256 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.21881 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.09939 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.947493 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.14557 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.92066 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.02924 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.02195 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.00054 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.876645 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.26341 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.791413 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.21982 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.863067 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.17524 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.02087 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.947546 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.04826 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.16324 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.19226 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.1496 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.808412 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.969894 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.977989 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.08571 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.03152 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.4449 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.913639 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.21213 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.697913 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.937949 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.794115 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.93009 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.16552 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.800843 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.23628 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.11019 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.04913 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.01376 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.23355 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.59700052038453 z_k1_s0000
                                 - 0.410001993547444 z_k1_s0001
                                 - 0.269000439949934 z_k1_s0002
                                 - 0.48100100905149 z_k1_s0003
                                 - 0.421999854163469 z_k1_s0004
                                 - 0.758000527546561 z_k1_s0005
                                 - 0.415999030851257 z_k1_s0006
                                 - 0.433001994570591 z_k1_s0007
                                 - 0.203000342276649 z_k1_s0010
                                 - 0.118000171591328 z_k1_s0011
                                 - 0.115000144927122 z_k1_s0012
                                 - 0.1950002711721 z_k1_s0013
                                 - 0.177000111186863 z_k1_s0014
                                 - 0.286999626873122 z_k1_s0015
                                 - 0.167000022306176 z_k1_s0016
                                 - 0.200000315612443 z_k1_s0017
                                 - 0.085999993899863 z_k1_s0020
                                 - 0.103999843892885 z_k1_s0021
                                 - 0.078000028269637 z_k1_s0022
                                 - 0.172000375399655 z_k1_s0023
                                 - 0.135000086197442 z_k1_s0024
                                 - 0.211999719050874 z_k1_s0025
                                 - 0.084999986083587 z_k1_s0026
                                 - 0.0969997891789527 z_k1_s0027
                                 - 0.344000255174588 z_k1_s0030
                                 - 0.17899940336101 z_k1_s0031
                                 - 0.205000297950009 z_k1_s0032
                                 - 0.104000093355112 z_k1_s0033
                                 - 0.0810000633265752 z_k1_s0034
                                 - 0.313999973572087 z_k1_s0035
                                 - 0.0940000924209686 z_k1_s0036
                                 - 0.0730000346992535 z_k1_s0037
                                 - 0.316001057747206 z_k1_s0040
                                 - 0.134000105094937 z_k1_s0041
                                 - 0.270000840505451 z_k1_s0042
                                 - 0.124000095120685 z_k1_s0043
                                 - 0.0780000206790954 z_k1_s0044
                                 - 0.210000495059611 z_k1_s0045
                                 - 0.117000173818808 z_k1_s0046
                                 - 0.179000007178903 z_k1_s0047
                                 - 0.248999723116538 z_k1_s0050
                                 - 0.186000374254241 z_k1_s0051
                                 - 0.186000374254241 z_k1_s0052
                                 - 0.186000374254241 z_k1_s0053
                                 - 0.107999678761905 z_k1_s0054
                                 - 0.171999612064939 z_k1_s0055
                                 - 0.486999324368834 z_k1_s0056
                                 - 0.280998391271377 z_k1_s0057
                                 - 0.140000373200347 z_k1_s0060
                                 - 0.276000905438334 z_k1_s0061
                                 - 0.271001528486654 z_k1_s0062
                                 - 0.195001156191672 z_k1_s0063
                                 - 0.20200130212707 z_k1_s0064
                                 - 0.249001554691139 z_k1_s0065
                                 - 0.0870001894922316 z_k1_s0066
                                 - 0.277000441427656 z_k1_s0067
                                 - 0.157000403705953 z_k1_s0070
                                 - 0.716002937772655 z_k1_s0071
                                 - 0.541002550664272 z_k1_s0072
                                 - 0.749000867021841 z_k1_s0073
                                 - 0.41600058172534 z_k1_s0074
                                 - 0.256000509869832 z_k1_s0075
                                 - 0.669001818346638 z_k1_s0076
                                 - 0.445002112649947 z_k1_s0077
                                 - 0.261000072121091 z_k1_s0100
                                 - 0.219999391060559 z_k1_s0101
                                 - 0.209999000558204 z_k1_s0102
                                 - 0.137999868937749 z_k1_s0103
                                 - 0.267999425473613 z_k1_s0104
                                 - 0.103999829228517 z_k1_s0105
                                 - 0.409000331561195 z_k1_s0106
                                 - 0.229999781562915 z_k1_s0107
                                 - 0.309001863019707 z_k1_s0110
                                 - 0.565000490048033 z_k1_s0111
                                 - 0.192000405273302 z_k1_s0112
                                 - 0.287999172907896 z_k1_s0113
                                 - 0.203000270043941 z_k1_s0114
                                 - 0.646000616178893 z_k1_s0115
                                 - 0.370000930474807 z_k1_s0116
                                 - 0.296999505811782 z_k1_s0117
                                 - 0.226999590560356 z_k1_s0120
                                 - 0.189999700433725 z_k1_s0121
                                 - 0.291000599210341 z_k1_s0122
                                 - 0.309999935654499 z_k1_s0123
                                 - 0.198999689275319 z_k1_s0124
                                 - 0.505998092651191 z_k1_s0125
                                 - 0.91299918804128 z_k1_s0126
                                 - 0.243999633483287 z_k1_s0127
                                 - 0.0789998458701085 z_k1_s0130
                                 - 0.113999613615327 z_k1_s0131
                                 - 0.187999984206847 z_k1_s0132
                                 - 0.054999895256401 z_k1_s0133
                                 - 0.13399974112651 z_k1_s0134
                                 - 0.17099942042299 z_k1_s0135
                                 - 0.0969998594303171 z_k1_s0136
                                 - 0.173999996148875 z_k1_s0137
                                 - 0.139000358786451 z_k1_s0140
                                 - 0.162000429444673 z_k1_s0141
                                 - 0.232000576256459 z_k1_s0142
                                 - 0.42000177767326 z_k1_s0143
                                 - 0.263000798213552 z_k1_s0144
                                 - 0.464001152512918 z_k1_s0145
                                 - 0.201000578500002 z_k1_s0146
                                 - 0.361001138270434 z_k1_s0147
                                 - 0.263999796117456 z_k1_s0150
                                 - 0.151000235064087 z_k1_s0151
                                 - 0.171000374542988 z_k1_s0152
                                 - 0.148000214142252 z_k1_s0153
                                 - 0.180000437308494 z_k1_s0154
                                 - 0.309000109944983 z_k1_s0155
                                 - 0.150000228090142 z_k1_s0156
                                 - 0.672000596482998 z_k1_s0157
                                 - 0.504998786638111 z_k1_s0160
                                 - 0.923998191887548 z_k1_s0161
                                 - 0.319999420647545 z_k1_s0162
                                 - 0.335999266850164 z_k1_s0163
                                 - 0.229999661609022 z_k1_s0164
                                 - 1.06699900184422 z_k1_s0165
                                 - 0.559998778083607 z_k1_s0166
                                 - 0.218999663319923 z_k1_s0167
                                 - 0.261001692278815 z_k1_s0170
                                 - 0.251999766676854 z_k1_s0171
                                 - 0.226999712466594 z_k1_s0172
                                 - 0.263000916853072 z_k1_s0173
                                 - 0.14200039455456 z_k1_s0174
                                 - 0.164000528899873 z_k1_s0175
                                 - 0.380002123600221 z_k1_s0176
                                 - 0.477002420576314 z_k1_s0177
                                 - 0.385998630813178 z_k1_s1000
                                 - 0.369999682509812 z_k1_s1001
                                 - 0.669999013193538 z_k1_s1002
                                 - 0.284999821349434 z_k1_s1003
                                 - 0.45299975133225 z_k1_s1004
                                 - 0.283000029011496 z_k1_s1005
                                 - 0.336998994134778 z_k1_s1006
                                 - 0.264000020601538 z_k1_s1007
                                 - 0.502998846763527 z_k1_s1010
                                 - 0.610996134533227 z_k1_s1011
                                 - 0.466999033108465 z_k1_s1012
                                 - 0.613999842238346 z_k1_s1013
                                 - 1.30999875163096 z_k1_s1014
                                 - 0.490999088311464 z_k1_s1015
                                 - 0.752999376936527 z_k1_s1016
                                 - 0.633997645328879 z_k1_s1017
                                 - 1.30899354415198 z_k1_s1020
                                 - 1.50499325902811 z_k1_s1021
                                 - 1.07599406043642 z_k1_s1022
                                 - 0.373998900283822 z_k1_s1023
                                 - 0.419998644208434 z_k1_s1024
                                 - 0.30599915289568 z_k1_s1025
                                 - 0.457998281546656 z_k1_s1026
                                 - 0.344998658732679 z_k1_s1027
                                 - 1.11299471591572 z_k1_s1030
                                 - 1.02900003419474 z_k1_s1031
                                 - 0.909999524549868 z_k1_s1032
                                 - 0.795999530465609 z_k1_s1033
                                 - 1.12200385422329 z_k1_s1034
                                 - 1.33600417402175 z_k1_s1035
                                 - 1.55399869306987 z_k1_s1036
                                 - 3.23500182728101 z_k1_s1037
                                 - 0.483000689154786 z_k1_s1040
                                 - 0.44399803040591 z_k1_s1041
                                 - 0.579000887620268 z_k1_s1042
                                 - 0.254999640299876 z_k1_s1043
                                 - 0.849000002055965 z_k1_s1044
                                 - 0.183999885392698 z_k1_s1045
                                 - 0.228999627798764 z_k1_s1046
                                 - 0.276999727031505 z_k1_s1047
                                 - 0.170999949343751 z_k1_s1050
                                 - 0.201000002588283 z_k1_s1051
                                 - 0.156000040771512 z_k1_s1052
                                 - 0.407001076500894 z_k1_s1053
                                 - 0.168999945794116 z_k1_s1054
                                 - 0.243999960855419 z_k1_s1055
                                 - 0.548000028199901 z_k1_s1056
                                 - 0.464999054539844 z_k1_s1057
                                 - 0.29099983953455 z_k1_s1060
                                 - 0.863000765129702 z_k1_s1061
                                 - 0.439999340842703 z_k1_s1062
                                 - 0.2780000696687 z_k1_s1063
                                 - 0.298000072116089 z_k1_s1064
                                 - 0.429999517868972 z_k1_s1065
                                 - 0.485000326724129 z_k1_s1066
                                 - 0.479998632737625 z_k1_s1067
                                 - 0.777002014697764 z_k1_s1070
                                 - 0.731002793391065 z_k1_s1071
                                 - 1.61800013134482 z_k1_s1072
                                 - 0.95799941981746 z_k1_s1073
                                 - 0.775002361293031 z_k1_s1074
                                 - 0.503999693266597 z_k1_s1075
                                 - 1.1659993388989 z_k1_s1076
                                 - 0.66400001833682 z_k1_s1077
                                 - 0.581000667404778 z_k1_s1100
                                 - 0.212000511270449 z_k1_s1101
                                 - 0.907001493660362 z_k1_s1102
                                 - 0.25100048697562 z_k1_s1103
                                 - 0.5750022151456 z_k1_s1104
                                 - 0.182000529958779 z_k1_s1105
                                 - 0.280000726244094 z_k1_s1106
                                 - 0.87200280213271 z_k1_s1107
                                 - 0.237000399686149 z_k1_s1110
                                 - 0.258999583866738 z_k1_s1111
                                 - 0.221999915743317 z_k1_s1112
                                 - 0.133000286253921 z_k1_s1113
                                 - 0.129000157202499 z_k1_s1114
                                 - 0.322001235026076 z_k1_s1115
                                 - 0.114000245360504 z_k1_s1116
                                 - 0.49099982223861 z_k1_s1117
                                 - 0.567002114305838 z_k1_s1120
                                 - 0.892999407656061 z_k1_s1121
                                 - 1.25600049383092 z_k1_s1122
                                 - 0.40899945702165 z_k1_s1123
                                 - 0.358999689546003 z_k1_s1124
                                 - 0.370999948235893 z_k1_s1125
                                 - 1.14100175871632 z_k1_s1126
                                 - 0.620000261655377 z_k1_s1127
                                 - 0.155000047873776 z_k1_s1130
                                 - 0.323000043699515 z_k1_s1131
                                 - 0.249000148988234 z_k1_s1132
                                 - 0.248000821438451 z_k1_s1133
                                 - 0.350000508553907 z_k1_s1134
                                 - 0.347000456903419 z_k1_s1135
                                 - 0.137000151771075 z_k1_s1136
                                 - 0.260001028040403 z_k1_s1137
                                 - 0.0560000069857713 z_k1_s1140
                                 - 0.151000552240848 z_k1_s1141
                                 - 0.266999898239139 z_k1_s1142
                                 - 0.099000133682135 z_k1_s1143
                                 - 0.271999704273348 z_k1_s1144
                                 - 0.12500015066126 z_k1_s1145
                                 - 0.231999973998129 z_k1_s1146
                                 - 0.0800001015512124 z_k1_s1147
                                 - 0.144000162832941 z_k1_s1150
                                 - 0.193000851662809 z_k1_s1151
                                 - 0.446000824421776 z_k1_s1152
                                 - 0.103000060116564 z_k1_s1153
                                 - 0.141000206778133 z_k1_s1154
                                 - 0.253999741710364 z_k1_s1155
                                 - 0.202000395226924 z_k1_s1156
                                 - 0.140000113226428 z_k1_s1157
                                 - 0.939999669087968 z_k1_s1160
                                 - 0.690997460528618 z_k1_s1161
                                 - 0.265999615223186 z_k1_s1162
                                 - 0.23099973554635 z_k1_s1163
                                 - 0.717000296992901 z_k1_s1164
                                 - 0.554997902847237 z_k1_s1165
                                 - 0.704001085757574 z_k1_s1166
                                 - 0.309999627908114 z_k1_s1167
                                 - 0.0880002077204007 z_k1_s1170
                                 - 0.213001084398033 z_k1_s1171
                                 - 0.103000174121495 z_k1_s1172
                                 - 0.400001417366212 z_k1_s1173
                                 - 0.164000268819649 z_k1_s1174
                                 - 0.421000155825809 z_k1_s1175
                                 - 0.100000180841276 z_k1_s1176
                                 - 0.246000836980165 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.62802 w_k1_ADM_s000
                                 + 0.783131 w_k1_AXP_s000
                                 + 0.60952 w_k1_DIS_s000
                                 + 0.72934 w_k1_FDX_s000
                                 + 0.827529 w_k1_IBM_s000
                                 + 0.793205 w_k1_INTC_s000
                                 + 0.653036 w_k1_KO_s000
                                 + 0.782303 w_k1_MAT_s000
                                 + 0.737273 w_k1_MSFT_s000
                                 + 0.823137 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.805089 w_k1_ADM_s000
                                 + 0.914471 w_k1_AXP_s000
                                 + 0.863003 w_k1_DIS_s000
                                 + 0.913409 w_k1_FDX_s000
                                 + 0.76644 w_k1_IBM_s000
                                 + 0.914328 w_k1_INTC_s000
                                 + 0.789539 w_k1_KO_s000
                                 + 0.713559 w_k1_MAT_s000
                                 + 0.929999 w_k1_MSFT_s000
                                 + 0.866132 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.16993 w_k1_ADM_s000
                                 + 0.957937 w_k1_AXP_s000
                                 + 1.27103 w_k1_DIS_s000
                                 + 0.950329 w_k1_FDX_s000
                                 + 1.3626 w_k1_IBM_s000
                                 + 1.01017 w_k1_INTC_s000
                                 + 1.28889 w_k1_KO_s000 + 1.06433 w_k1_MAT_s000
                                 + 1.09449 w_k1_MSFT_s000
                                 + 1.01755 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.03936 w_k1_ADM_s000
                                 + 0.977637 w_k1_AXP_s000
                                 + 1.07165 w_k1_DIS_s000
                                 + 1.07678 w_k1_FDX_s000
                                 + 0.81765 w_k1_IBM_s000
                                 + 1.07744 w_k1_INTC_s000
                                 + 0.849381 w_k1_KO_s000
                                 + 0.585596 w_k1_MAT_s000
                                 + 1.02879 w_k1_MSFT_s000
                                 + 0.987393 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.0826 w_k1_ADM_s000
                                 + 1.60411 w_k1_AXP_s000
                                 + 1.62637 w_k1_DIS_s000
                                 + 1.27085 w_k1_FDX_s000
                                 + 1.50451 w_k1_IBM_s000
                                 + 1.43423 w_k1_INTC_s000
                                 + 1.46735 w_k1_KO_s000 + 1.31329 w_k1_MAT_s000
                                 + 1.14804 w_k1_MSFT_s000
                                 + 1.41361 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.824873 w_k1_ADM_s000
                                 + 1.09779 w_k1_AXP_s000
                                 + 0.947186 w_k1_DIS_s000
                                 + 0.904107 w_k1_FDX_s000 + 1.303 w_k1_IBM_s000
                                 + 0.914355 w_k1_INTC_s000
                                 + 1.00775 w_k1_KO_s000 + 1.43908 w_k1_MAT_s000
                                 + 0.855398 w_k1_MSFT_s000
                                 + 1.0616 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.786396 w_k1_ADM_s000
                                 + 0.805751 w_k1_AXP_s000
                                 + 0.892758 w_k1_DIS_s000
                                 + 0.802444 w_k1_FDX_s000
                                 + 0.870065 w_k1_IBM_s000
                                 + 0.895034 w_k1_INTC_s000
                                 + 0.919273 w_k1_KO_s000
                                 + 0.472526 w_k1_MAT_s000
                                 + 0.859 w_k1_MSFT_s000 + 0.854956 w_k1_PG_s000
                                 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.22443 w_k1_ADM_s000
                                 + 1.19634 w_k1_AXP_s000
                                 + 1.34607 w_k1_DIS_s000
                                 + 1.28292 w_k1_FDX_s000
                                 + 1.00522 w_k1_IBM_s000
                                 + 1.16452 w_k1_INTC_s000
                                 + 1.09821 w_k1_KO_s000 + 1.14835 w_k1_MAT_s000
                                 + 1.21162 w_k1_MSFT_s000
                                 + 1.10369 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.876184 w_k1_ADM_s001
                                 + 0.849041 w_k1_AXP_s001
                                 + 0.775454 w_k1_DIS_s001
                                 + 0.863693 w_k1_FDX_s001
                                 + 0.715606 w_k1_IBM_s001
                                 + 0.784115 w_k1_INTC_s001
                                 + 0.88105 w_k1_KO_s001
                                 + 0.935314 w_k1_MAT_s001
                                 + 0.82065 w_k1_MSFT_s001
                                 + 0.89419 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.780478 w_k1_ADM_s001
                                 + 0.727795 w_k1_AXP_s001
                                 + 0.53408 w_k1_DIS_s001
                                 + 0.63906 w_k1_FDX_s001
                                 + 0.786611 w_k1_IBM_s001
                                 + 0.778259 w_k1_INTC_s001
                                 + 0.759968 w_k1_KO_s001
                                 + 0.815395 w_k1_MAT_s001
                                 + 0.712003 w_k1_MSFT_s001
                                 + 0.788786 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.2351 w_k1_ADM_s001
                                 + 1.46026 w_k1_AXP_s001
                                 + 0.808847 w_k1_DIS_s001
                                 + 1.02458 w_k1_FDX_s001
                                 + 0.998422 w_k1_IBM_s001
                                 + 1.02516 w_k1_INTC_s001
                                 + 0.858299 w_k1_KO_s001
                                 + 0.963876 w_k1_MAT_s001
                                 + 0.787293 w_k1_MSFT_s001
                                 + 0.937654 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.02464 w_k1_ADM_s001
                                 + 0.924761 w_k1_AXP_s001
                                 + 1.06435 w_k1_DIS_s001
                                 + 0.899927 w_k1_FDX_s001
                                 + 1.41056 w_k1_IBM_s001
                                 + 1.00184 w_k1_INTC_s001
                                 + 1.48717 w_k1_KO_s001 + 1.11555 w_k1_MAT_s001
                                 + 1.01926 w_k1_MSFT_s001
                                 + 1.01069 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.46444 w_k1_ADM_s001
                                 + 1.30665 w_k1_AXP_s001
                                 + 1.45571 w_k1_DIS_s001
                                 + 1.32608 w_k1_FDX_s001
                                 + 0.987427 w_k1_IBM_s001
                                 + 1.38456 w_k1_INTC_s001
                                 + 1.21789 w_k1_KO_s001
                                 + 0.854488 w_k1_MAT_s001
                                 + 1.3361 w_k1_MSFT_s001 + 1.07533 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.0824 w_k1_ADM_s001
                                 + 0.90412 w_k1_AXP_s001
                                 + 1.17278 w_k1_DIS_s001
                                 + 1.04092 w_k1_FDX_s001
                                 + 0.869413 w_k1_IBM_s001
                                 + 1.15641 w_k1_INTC_s001
                                 + 1.11066 w_k1_KO_s001 + 0.65717 w_k1_MAT_s001
                                 + 1.08106 w_k1_MSFT_s001
                                 + 1.06436 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.02563 w_k1_ADM_s001
                                 + 1.21873 w_k1_AXP_s001 + 1.6776 w_k1_DIS_s001
                                 + 1.15541 w_k1_FDX_s001
                                 + 1.62533 w_k1_IBM_s001
                                 + 1.22521 w_k1_INTC_s001
                                 + 1.49692 w_k1_KO_s001 + 1.41799 w_k1_MAT_s001
                                 + 1.2148 w_k1_MSFT_s001 + 1.0507 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.940467 w_k1_ADM_s001
                                 + 0.793464 w_k1_AXP_s001
                                 + 0.914584 w_k1_DIS_s001
                                 + 0.770735 w_k1_FDX_s001
                                 + 0.961771 w_k1_IBM_s001
                                 + 0.687781 w_k1_INTC_s001
                                 + 1.06745 w_k1_KO_s001
                                 + 0.850001 w_k1_MAT_s001
                                 + 0.840205 w_k1_MSFT_s001
                                 + 0.961636 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.18613 w_k1_ADM_s002
                                 + 0.964396 w_k1_AXP_s002
                                 + 0.870471 w_k1_DIS_s002
                                 + 1.0315 w_k1_FDX_s002
                                 + 0.630115 w_k1_IBM_s002
                                 + 0.951907 w_k1_INTC_s002
                                 + 0.907919 w_k1_KO_s002
                                 + 0.726088 w_k1_MAT_s002
                                 + 0.841888 w_k1_MSFT_s002
                                 + 1.19024 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.775212 w_k1_ADM_s002
                                 + 1.29209 w_k1_AXP_s002
                                 + 0.959124 w_k1_DIS_s002
                                 + 0.91014 w_k1_FDX_s002
                                 + 1.49085 w_k1_IBM_s002
                                 + 0.92533 w_k1_INTC_s002
                                 + 1.10195 w_k1_KO_s002 + 1.79446 w_k1_MAT_s002
                                 + 0.832119 w_k1_MSFT_s002
                                 + 1.1783 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.632327 w_k1_ADM_s002
                                 + 0.762695 w_k1_AXP_s002
                                 + 1.09396 w_k1_DIS_s002
                                 + 1.01262 w_k1_FDX_s002
                                 + 0.803451 w_k1_IBM_s002
                                 + 1.21951 w_k1_INTC_s002
                                 + 0.955829 w_k1_KO_s002
                                 + 0.663502 w_k1_MAT_s002
                                 + 1.09039 w_k1_MSFT_s002
                                 + 0.918806 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.03556 w_k1_ADM_s002
                                 + 0.97242 w_k1_AXP_s002
                                 + 0.818814 w_k1_DIS_s002
                                 + 0.835219 w_k1_FDX_s002
                                 + 1.04898 w_k1_IBM_s002
                                 + 0.807833 w_k1_INTC_s002
                                 + 0.859046 w_k1_KO_s002 + 1.1418 w_k1_MAT_s002
                                 + 0.680735 w_k1_MSFT_s002
                                 + 0.971862 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.795247 w_k1_ADM_s002
                                 + 1.40437 w_k1_AXP_s002
                                 + 1.72106 w_k1_DIS_s002 + 1.2363 w_k1_FDX_s002
                                 + 1.36645 w_k1_IBM_s002
                                 + 1.34244 w_k1_INTC_s002
                                 + 1.30306 w_k1_KO_s002 + 1.19254 w_k1_MAT_s002
                                 + 1.14309 w_k1_MSFT_s002
                                 + 1.15272 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.991593 w_k1_ADM_s002
                                 + 0.689138 w_k1_AXP_s002
                                 + 0.674997 w_k1_DIS_s002
                                 + 0.780091 w_k1_FDX_s002
                                 + 0.852679 w_k1_IBM_s002
                                 + 0.865423 w_k1_INTC_s002
                                 + 0.78831 w_k1_KO_s002
                                 + 0.726984 w_k1_MAT_s002
                                 + 0.720736 w_k1_MSFT_s002
                                 + 0.89741 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.16183 w_k1_ADM_s002
                                 + 1.14626 w_k1_AXP_s002
                                 + 1.28491 w_k1_DIS_s002
                                 + 1.25234 w_k1_FDX_s002
                                 + 1.00251 w_k1_IBM_s002
                                 + 1.25623 w_k1_INTC_s002
                                 + 1.16699 w_k1_KO_s002
                                 + 0.777391 w_k1_MAT_s002
                                 + 1.03492 w_k1_MSFT_s002
                                 + 1.14562 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.01258 w_k1_ADM_s002
                                 + 1.20382 w_k1_AXP_s002
                                 + 1.12756 w_k1_DIS_s002
                                 + 1.21516 w_k1_FDX_s002
                                 + 0.916295 w_k1_IBM_s002
                                 + 1.10409 w_k1_INTC_s002
                                 + 1.15668 w_k1_KO_s002 + 1.2136 w_k1_MAT_s002
                                 + 1.01681 w_k1_MSFT_s002
                                 + 1.03461 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.1022 w_k1_ADM_s003
                                 + 1.20865 w_k1_AXP_s003
                                 + 1.17901 w_k1_DIS_s003
                                 + 1.08888 w_k1_FDX_s003
                                 + 1.08316 w_k1_IBM_s003
                                 + 0.979643 w_k1_INTC_s003
                                 + 1.07569 w_k1_KO_s003 + 1.30851 w_k1_MAT_s003
                                 + 1.05262 w_k1_MSFT_s003
                                 + 1.04051 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.972464 w_k1_ADM_s003
                                 + 1.08741 w_k1_AXP_s003
                                 + 1.16274 w_k1_DIS_s003
                                 + 1.13887 w_k1_FDX_s003
                                 + 0.900994 w_k1_IBM_s003
                                 + 1.19835 w_k1_INTC_s003
                                 + 1.07643 w_k1_KO_s003
                                 + 0.864157 w_k1_MAT_s003
                                 + 1.26194 w_k1_MSFT_s003
                                 + 0.982713 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.06843 w_k1_ADM_s003
                                 + 1.18117 w_k1_AXP_s003
                                 + 1.68519 w_k1_DIS_s003
                                 + 1.28343 w_k1_FDX_s003
                                 + 1.37457 w_k1_IBM_s003
                                 + 1.23314 w_k1_INTC_s003
                                 + 1.50828 w_k1_KO_s003 + 1.5451 w_k1_MAT_s003
                                 + 1.2623 w_k1_MSFT_s003 + 1.25294 w_k1_PG_s003
                                 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.04061 w_k1_ADM_s003
                                 + 1.06875 w_k1_AXP_s003
                                 + 0.882774 w_k1_DIS_s003
                                 + 1.04458 w_k1_FDX_s003
                                 + 0.625389 w_k1_IBM_s003
                                 + 0.90696 w_k1_INTC_s003
                                 + 0.74077 w_k1_KO_s003
                                 + 0.770962 w_k1_MAT_s003
                                 + 0.882903 w_k1_MSFT_s003
                                 + 0.85053 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.561862 w_k1_ADM_s003
                                 + 0.721007 w_k1_AXP_s003
                                 + 0.662545 w_k1_DIS_s003
                                 + 0.756722 w_k1_FDX_s003
                                 + 0.757324 w_k1_IBM_s003
                                 + 0.72275 w_k1_INTC_s003
                                 + 0.734345 w_k1_KO_s003
                                 + 1.03178 w_k1_MAT_s003
                                 + 0.802052 w_k1_MSFT_s003
                                 + 0.834305 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.840608 w_k1_ADM_s003
                                 + 0.791762 w_k1_AXP_s003
                                 + 1.09541 w_k1_DIS_s003
                                 + 0.883262 w_k1_FDX_s003
                                 + 0.797579 w_k1_IBM_s003
                                 + 0.928517 w_k1_INTC_s003
                                 + 1.02886 w_k1_KO_s003
                                 + 0.731294 w_k1_MAT_s003
                                 + 1.06092 w_k1_MSFT_s003
                                 + 0.993673 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.6453 w_k1_ADM_s003
                                 + 0.475533 w_k1_AXP_s003
                                 + 0.580636 w_k1_DIS_s003
                                 + 0.66392 w_k1_FDX_s003
                                 + 0.664678 w_k1_IBM_s003
                                 + 0.639017 w_k1_INTC_s003
                                 + 0.683282 w_k1_KO_s003
                                 + 0.557685 w_k1_MAT_s003
                                 + 0.760172 w_k1_MSFT_s003
                                 + 0.745302 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.971936 w_k1_ADM_s003
                                 + 0.823879 w_k1_AXP_s003
                                 + 0.898788 w_k1_DIS_s003
                                 + 0.833483 w_k1_FDX_s003
                                 + 1.28219 w_k1_IBM_s003
                                 + 0.81103 w_k1_INTC_s003
                                 + 0.909105 w_k1_KO_s003
                                 + 1.38219 w_k1_MAT_s003
                                 + 0.815795 w_k1_MSFT_s003
                                 + 0.969597 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.07673 w_k1_ADM_s004
                                 + 1.02775 w_k1_AXP_s004
                                 + 0.867244 w_k1_DIS_s004
                                 + 0.886432 w_k1_FDX_s004
                                 + 1.06339 w_k1_IBM_s004
                                 + 0.914297 w_k1_INTC_s004
                                 + 1.06193 w_k1_KO_s004 + 1.24546 w_k1_MAT_s004
                                 + 0.912424 w_k1_MSFT_s004
                                 + 0.992465 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.909483 w_k1_ADM_s004
                                 + 0.679528 w_k1_AXP_s004
                                 + 0.885395 w_k1_DIS_s004
                                 + 0.992938 w_k1_FDX_s004
                                 + 0.489519 w_k1_IBM_s004
                                 + 1.07603 w_k1_INTC_s004
                                 + 0.746671 w_k1_KO_s004
                                 + 0.442827 w_k1_MAT_s004
                                 + 0.995256 w_k1_MSFT_s004
                                 + 0.83734 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.18744 w_k1_ADM_s004
                                 + 0.875053 w_k1_AXP_s004
                                 + 1.09565 w_k1_DIS_s004 + 1.0633 w_k1_FDX_s004
                                 + 0.894201 w_k1_IBM_s004
                                 + 1.06903 w_k1_INTC_s004
                                 + 1.07879 w_k1_KO_s004
                                 + 0.788676 w_k1_MAT_s004
                                 + 1.04466 w_k1_MSFT_s004
                                 + 0.987797 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.3033 w_k1_ADM_s004
                                 + 1.55748 w_k1_AXP_s004
                                 + 1.27033 w_k1_DIS_s004
                                 + 1.34135 w_k1_FDX_s004
                                 + 0.784888 w_k1_IBM_s004
                                 + 1.22778 w_k1_INTC_s004
                                 + 1.01575 w_k1_KO_s004 + 1.16657 w_k1_MAT_s004
                                 + 1.10868 w_k1_MSFT_s004
                                 + 1.11942 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.15872 w_k1_ADM_s004
                                 + 1.39923 w_k1_AXP_s004
                                 + 1.80171 w_k1_DIS_s004
                                 + 1.42506 w_k1_FDX_s004 + 1.474 w_k1_IBM_s004
                                 + 1.16073 w_k1_INTC_s004
                                 + 1.46492 w_k1_KO_s004 + 1.38125 w_k1_MAT_s004
                                 + 1.17311 w_k1_MSFT_s004
                                 + 1.12598 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.734933 w_k1_ADM_s004
                                 + 0.653221 w_k1_AXP_s004
                                 + 0.6094 w_k1_DIS_s004 + 0.71718 w_k1_FDX_s004
                                 + 0.904528 w_k1_IBM_s004
                                 + 0.857205 w_k1_INTC_s004
                                 + 0.804198 w_k1_KO_s004
                                 + 0.852316 w_k1_MAT_s004
                                 + 0.715546 w_k1_MSFT_s004
                                 + 0.896521 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.889243 w_k1_ADM_s004
                                 + 0.991459 w_k1_AXP_s004
                                 + 1.56278 w_k1_DIS_s004
                                 + 1.20091 w_k1_FDX_s004
                                 + 0.993311 w_k1_IBM_s004
                                 + 1.33567 w_k1_INTC_s004 + 1.2527 w_k1_KO_s004
                                 + 1.16011 w_k1_MAT_s004
                                 + 1.29558 w_k1_MSFT_s004
                                 + 1.01243 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.953597 w_k1_ADM_s004
                                 + 1.14473 w_k1_AXP_s004 + 1.0984 w_k1_DIS_s004
                                 + 0.994626 w_k1_FDX_s004
                                 + 1.41973 w_k1_IBM_s004
                                 + 1.0293 w_k1_INTC_s004 + 1.23472 w_k1_KO_s004
                                 + 1.54237 w_k1_MAT_s004
                                 + 1.04912 w_k1_MSFT_s004
                                 + 1.12168 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.20136 w_k1_ADM_s005
                                 + 0.922962 w_k1_AXP_s005
                                 + 1.08219 w_k1_DIS_s005
                                 + 1.09099 w_k1_FDX_s005
                                 + 0.806727 w_k1_IBM_s005
                                 + 1.26304 w_k1_INTC_s005
                                 + 1.09357 w_k1_KO_s005 + 0.89233 w_k1_MAT_s005
                                 + 1.23619 w_k1_MSFT_s005
                                 + 1.09576 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.905159 w_k1_ADM_s005
                                 + 1.11287 w_k1_AXP_s005
                                 + 0.734288 w_k1_DIS_s005
                                 + 0.787321 w_k1_FDX_s005
                                 + 1.02114 w_k1_IBM_s005
                                 + 0.886532 w_k1_INTC_s005
                                 + 0.967789 w_k1_KO_s005
                                 + 1.19701 w_k1_MAT_s005
                                 + 0.864499 w_k1_MSFT_s005
                                 + 0.952548 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.847925 w_k1_ADM_s005
                                 + 0.814235 w_k1_AXP_s005
                                 + 0.652037 w_k1_DIS_s005
                                 + 0.828161 w_k1_FDX_s005
                                 + 0.616339 w_k1_IBM_s005
                                 + 0.760269 w_k1_INTC_s005
                                 + 0.731979 w_k1_KO_s005
                                 + 0.85766 w_k1_MAT_s005
                                 + 0.906369 w_k1_MSFT_s005
                                 + 0.811804 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.39771 w_k1_ADM_s005
                                 + 1.38215 w_k1_AXP_s005
                                 + 1.26796 w_k1_DIS_s005
                                 + 1.02688 w_k1_FDX_s005
                                 + 1.57764 w_k1_IBM_s005
                                 + 0.82019 w_k1_INTC_s005
                                 + 1.10672 w_k1_KO_s005 + 1.54823 w_k1_MAT_s005
                                 + 0.970435 w_k1_MSFT_s005
                                 + 1.02413 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.844107 w_k1_ADM_s005
                                 + 1.10752 w_k1_AXP_s005
                                 + 0.55394 w_k1_DIS_s005
                                 + 0.68749 w_k1_FDX_s005 + 1.2175 w_k1_IBM_s005
                                 + 0.678213 w_k1_INTC_s005
                                 + 0.843278 w_k1_KO_s005
                                 + 1.45034 w_k1_MAT_s005
                                 + 0.644236 w_k1_MSFT_s005
                                 + 0.956683 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.777284 w_k1_ADM_s005
                                 + 0.712967 w_k1_AXP_s005
                                 + 0.821069 w_k1_DIS_s005
                                 + 0.777569 w_k1_FDX_s005
                                 + 0.825633 w_k1_IBM_s005
                                 + 0.778964 w_k1_INTC_s005
                                 + 0.720904 w_k1_KO_s005
                                 + 0.785501 w_k1_MAT_s005
                                 + 0.808619 w_k1_MSFT_s005
                                 + 0.899555 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.935775 w_k1_ADM_s005
                                 + 1.32871 w_k1_AXP_s005
                                 + 1.16041 w_k1_DIS_s005 + 1.1168 w_k1_FDX_s005
                                 + 1.14073 w_k1_IBM_s005
                                 + 0.965547 w_k1_INTC_s005
                                 + 1.11549 w_k1_KO_s005 + 1.46589 w_k1_MAT_s005
                                 + 1.16479 w_k1_MSFT_s005
                                 + 1.09035 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.60565 w_k1_ADM_s005
                                 + 1.27073 w_k1_AXP_s005
                                 + 1.16137 w_k1_DIS_s005 + 1.2185 w_k1_FDX_s005
                                 + 0.868735 w_k1_IBM_s005
                                 + 1.04309 w_k1_INTC_s005
                                 + 0.903326 w_k1_KO_s005
                                 + 1.01884 w_k1_MAT_s005
                                 + 1.14793 w_k1_MSFT_s005
                                 + 1.05684 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.895938 w_k1_ADM_s006
                                 + 0.551184 w_k1_AXP_s006
                                 + 0.663012 w_k1_DIS_s006
                                 + 0.796409 w_k1_FDX_s006
                                 + 0.739778 w_k1_IBM_s006
                                 + 1.00001 w_k1_INTC_s006
                                 + 0.788679 w_k1_KO_s006
                                 + 0.623041 w_k1_MAT_s006
                                 + 1.17355 w_k1_MSFT_s006
                                 + 0.724901 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.10856 w_k1_ADM_s006
                                 + 1.29653 w_k1_AXP_s006
                                 + 0.914418 w_k1_DIS_s006
                                 + 1.01761 w_k1_FDX_s006
                                 + 1.06223 w_k1_IBM_s006
                                 + 0.924278 w_k1_INTC_s006
                                 + 0.919702 w_k1_KO_s006
                                 + 1.23489 w_k1_MAT_s006
                                 + 1.01491 w_k1_MSFT_s006
                                 + 0.87284 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.08827 w_k1_ADM_s006
                                 + 0.948437 w_k1_AXP_s006
                                 + 0.717994 w_k1_DIS_s006
                                 + 0.895825 w_k1_FDX_s006
                                 + 0.714771 w_k1_IBM_s006
                                 + 1.1948 w_k1_INTC_s006
                                 + 0.783548 w_k1_KO_s006
                                 + 0.637813 w_k1_MAT_s006
                                 + 1.18045 w_k1_MSFT_s006
                                 + 0.954137 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.05573 w_k1_ADM_s006
                                 + 0.756937 w_k1_AXP_s006
                                 + 0.67771 w_k1_DIS_s006
                                 + 0.812191 w_k1_FDX_s006
                                 + 0.995032 w_k1_IBM_s006
                                 + 0.918497 w_k1_INTC_s006
                                 + 0.921113 w_k1_KO_s006
                                 + 1.04055 w_k1_MAT_s006
                                 + 1.05791 w_k1_MSFT_s006
                                 + 0.898057 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.912521 w_k1_ADM_s006
                                 + 1.07716 w_k1_AXP_s006 + 1.158 w_k1_DIS_s006
                                 + 1.08214 w_k1_FDX_s006
                                 + 0.78017 w_k1_IBM_s006
                                 + 1.50156 w_k1_INTC_s006
                                 + 0.991183 w_k1_KO_s006
                                 + 0.649396 w_k1_MAT_s006
                                 + 1.3697 w_k1_MSFT_s006
                                 + 0.943118 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.34562 w_k1_ADM_s006
                                 + 1.25383 w_k1_AXP_s006
                                 + 1.27134 w_k1_DIS_s006
                                 + 1.24196 w_k1_FDX_s006
                                 + 1.04297 w_k1_IBM_s006
                                 + 1.13944 w_k1_INTC_s006
                                 + 1.39036 w_k1_KO_s006 + 0.92947 w_k1_MAT_s006
                                 + 1.51489 w_k1_MSFT_s006
                                 + 1.27261 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.773464 w_k1_ADM_s006
                                 + 1.01543 w_k1_AXP_s006
                                 + 0.775393 w_k1_DIS_s006
                                 + 0.836569 w_k1_FDX_s006
                                 + 1.39373 w_k1_IBM_s006
                                 + 1.15091 w_k1_INTC_s006
                                 + 0.917194 w_k1_KO_s006
                                 + 1.38514 w_k1_MAT_s006
                                 + 0.970483 w_k1_MSFT_s006
                                 + 0.836206 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.06979 w_k1_ADM_s006
                                 + 1.5449 w_k1_AXP_s006 + 1.30127 w_k1_DIS_s006
                                 + 1.17779 w_k1_FDX_s006 + 1.5182 w_k1_IBM_s006
                                 + 1.30427 w_k1_INTC_s006 + 1.2635 w_k1_KO_s006
                                 + 1.6958 w_k1_MAT_s006
                                 + 1.22569 w_k1_MSFT_s006
                                 + 1.13601 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.26633 w_k1_ADM_s007
                                 + 1.72434 w_k1_AXP_s007
                                 + 1.16014 w_k1_DIS_s007
                                 + 1.20914 w_k1_FDX_s007
                                 + 1.11955 w_k1_IBM_s007
                                 + 1.32994 w_k1_INTC_s007
                                 + 1.07256 w_k1_KO_s007 + 1.5492 w_k1_MAT_s007
                                 + 0.995407 w_k1_MSFT_s007
                                 + 1.11616 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.15667 w_k1_ADM_s007
                                 + 1.0297 w_k1_AXP_s007 + 1.21578 w_k1_DIS_s007
                                 + 1.09636 w_k1_FDX_s007
                                 + 0.809684 w_k1_IBM_s007
                                 + 1.06379 w_k1_INTC_s007
                                 + 0.978306 w_k1_KO_s007
                                 + 0.581385 w_k1_MAT_s007
                                 + 1.17614 w_k1_MSFT_s007
                                 + 0.930399 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.12428 w_k1_ADM_s007
                                 + 1.00888 w_k1_AXP_s007
                                 + 1.27156 w_k1_DIS_s007
                                 + 1.04125 w_k1_FDX_s007
                                 + 1.07969 w_k1_IBM_s007
                                 + 1.21079 w_k1_INTC_s007
                                 + 1.16555 w_k1_KO_s007
                                 + 0.961115 w_k1_MAT_s007
                                 + 1.14678 w_k1_MSFT_s007
                                 + 0.967413 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.01525 w_k1_ADM_s007
                                 + 0.996134 w_k1_AXP_s007
                                 + 0.75707 w_k1_DIS_s007
                                 + 0.921602 w_k1_FDX_s007
                                 + 1.07319 w_k1_IBM_s007
                                 + 1.18648 w_k1_INTC_s007
                                 + 0.91697 w_k1_KO_s007 + 1.07709 w_k1_MAT_s007
                                 + 0.938228 w_k1_MSFT_s007
                                 + 0.883352 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.828474 w_k1_ADM_s007
                                 + 0.642665 w_k1_AXP_s007
                                 + 0.639842 w_k1_DIS_s007
                                 + 0.79985 w_k1_FDX_s007
                                 + 0.635245 w_k1_IBM_s007
                                 + 0.771168 w_k1_INTC_s007
                                 + 0.70552 w_k1_KO_s007 + 0.61272 w_k1_MAT_s007
                                 + 0.74961 w_k1_MSFT_s007
                                 + 0.77075 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.5211 w_k1_ADM_s007
                                 + 1.54716 w_k1_AXP_s007
                                 + 1.93759 w_k1_DIS_s007
                                 + 1.47241 w_k1_FDX_s007
                                 + 1.18633 w_k1_IBM_s007
                                 + 1.57573 w_k1_INTC_s007
                                 + 1.41336 w_k1_KO_s007 + 0.92236 w_k1_MAT_s007
                                 + 1.52203 w_k1_MSFT_s007
                                 + 1.20671 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.97318 w_k1_ADM_s007
                                 + 0.697396 w_k1_AXP_s007
                                 + 0.812733 w_k1_DIS_s007
                                 + 0.761483 w_k1_FDX_s007
                                 + 1.03846 w_k1_IBM_s007
                                 + 1.04562 w_k1_INTC_s007
                                 + 1.11411 w_k1_KO_s007
                                 + 0.708376 w_k1_MAT_s007
                                 + 0.924595 w_k1_MSFT_s007
                                 + 0.904775 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.29144 w_k1_ADM_s007
                                 + 1.17143 w_k1_AXP_s007
                                 + 1.22199 w_k1_DIS_s007
                                 + 1.08413 w_k1_FDX_s007
                                 + 1.53307 w_k1_IBM_s007
                                 + 1.05372 w_k1_INTC_s007
                                 + 1.40637 w_k1_KO_s007 + 1.4182 w_k1_MAT_s007
                                 + 1.10107 w_k1_MSFT_s007
                                 + 1.00509 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.00966 w_k1_ADM_s010
                                 + 1.08866 w_k1_AXP_s010
                                 + 1.04201 w_k1_DIS_s010 + 1.0275 w_k1_FDX_s010
                                 + 1.0667 w_k1_IBM_s010
                                 + 1.11657 w_k1_INTC_s010
                                 + 0.948816 w_k1_KO_s010
                                 + 0.927111 w_k1_MAT_s010
                                 + 0.900926 w_k1_MSFT_s010
                                 + 1.0108 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.14119 w_k1_ADM_s010
                                 + 1.23971 w_k1_AXP_s010
                                 + 1.29098 w_k1_DIS_s010
                                 + 1.35678 w_k1_FDX_s010
                                 + 0.954852 w_k1_IBM_s010
                                 + 1.19796 w_k1_INTC_s010
                                 + 0.997868 w_k1_KO_s010
                                 + 0.811042 w_k1_MAT_s010
                                 + 1.29549 w_k1_MSFT_s010
                                 + 1.31951 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.58593 w_k1_ADM_s010
                                 + 0.667241 w_k1_AXP_s010
                                 + 0.538398 w_k1_DIS_s010
                                 + 0.704815 w_k1_FDX_s010
                                 + 0.60147 w_k1_IBM_s010
                                 + 0.867714 w_k1_INTC_s010
                                 + 0.603853 w_k1_KO_s010
                                 + 0.80706 w_k1_MAT_s010
                                 + 0.745997 w_k1_MSFT_s010
                                 + 0.794941 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.08332 w_k1_ADM_s010
                                 + 0.940591 w_k1_AXP_s010
                                 + 0.9571 w_k1_DIS_s010
                                 + 0.967758 w_k1_FDX_s010
                                 + 1.25441 w_k1_IBM_s010
                                 + 0.928101 w_k1_INTC_s010
                                 + 0.949669 w_k1_KO_s010
                                 + 1.04775 w_k1_MAT_s010
                                 + 0.840807 w_k1_MSFT_s010
                                 + 1.15433 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.733287 w_k1_ADM_s010
                                 + 0.858126 w_k1_AXP_s010
                                 + 0.791674 w_k1_DIS_s010
                                 + 0.793869 w_k1_FDX_s010
                                 + 1.01108 w_k1_IBM_s010
                                 + 1.08609 w_k1_INTC_s010
                                 + 0.852658 w_k1_KO_s010
                                 + 0.888604 w_k1_MAT_s010
                                 + 0.922037 w_k1_MSFT_s010
                                 + 0.933655 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.23651 w_k1_ADM_s010
                                 + 1.38622 w_k1_AXP_s010
                                 + 1.46639 w_k1_DIS_s010
                                 + 1.22386 w_k1_FDX_s010
                                 + 1.58926 w_k1_IBM_s010
                                 + 1.16679 w_k1_INTC_s010
                                 + 1.36409 w_k1_KO_s010 + 1.53901 w_k1_MAT_s010
                                 + 1.17689 w_k1_MSFT_s010
                                 + 1.25814 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.994168 w_k1_ADM_s010
                                 + 0.780778 w_k1_AXP_s010
                                 + 0.879016 w_k1_DIS_s010
                                 + 0.94897 w_k1_FDX_s010
                                 + 0.745805 w_k1_IBM_s010
                                 + 0.979566 w_k1_INTC_s010
                                 + 0.793237 w_k1_KO_s010
                                 + 0.773703 w_k1_MAT_s010
                                 + 0.897919 w_k1_MSFT_s010
                                 + 1.08109 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.06779 w_k1_ADM_s010
                                 + 1.19869 w_k1_AXP_s010
                                 + 1.29291 w_k1_DIS_s010
                                 + 1.11315 w_k1_FDX_s010 + 1.1681 w_k1_IBM_s010
                                 + 1.16192 w_k1_INTC_s010
                                 + 1.20092 w_k1_KO_s010 + 1.47485 w_k1_MAT_s010
                                 + 1.28565 w_k1_MSFT_s010
                                 + 1.25531 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.05209 w_k1_ADM_s011
                                 + 1.0566 w_k1_AXP_s011 + 1.01082 w_k1_DIS_s011
                                 + 0.87122 w_k1_FDX_s011
                                 + 1.34485 w_k1_IBM_s011
                                 + 1.01186 w_k1_INTC_s011 + 1.1049 w_k1_KO_s011
                                 + 1.33407 w_k1_MAT_s011
                                 + 0.818387 w_k1_MSFT_s011
                                 + 0.932412 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.08708 w_k1_ADM_s011
                                 + 1.1724 w_k1_AXP_s011 + 1.27315 w_k1_DIS_s011
                                 + 1.16472 w_k1_FDX_s011
                                 + 1.10514 w_k1_IBM_s011
                                 + 1.21998 w_k1_INTC_s011
                                 + 1.05631 w_k1_KO_s011
                                 + 0.910986 w_k1_MAT_s011
                                 + 1.05985 w_k1_MSFT_s011
                                 + 0.973572 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.656919 w_k1_ADM_s011
                                 + 0.723014 w_k1_AXP_s011
                                 + 0.702749 w_k1_DIS_s011
                                 + 0.812915 w_k1_FDX_s011
                                 + 1.12525 w_k1_IBM_s011
                                 + 0.974722 w_k1_INTC_s011
                                 + 0.84469 w_k1_KO_s011 + 1.27523 w_k1_MAT_s011
                                 + 0.746219 w_k1_MSFT_s011
                                 + 0.834296 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.12327 w_k1_ADM_s011
                                 + 1.08641 w_k1_AXP_s011
                                 + 1.25039 w_k1_DIS_s011
                                 + 1.19691 w_k1_FDX_s011
                                 + 0.868591 w_k1_IBM_s011
                                 + 1.43837 w_k1_INTC_s011 + 1.1642 w_k1_KO_s011
                                 + 1.0857 w_k1_MAT_s011
                                 + 1.16528 w_k1_MSFT_s011
                                 + 1.07119 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.23818 w_k1_ADM_s011
                                 + 0.884563 w_k1_AXP_s011
                                 + 0.871383 w_k1_DIS_s011
                                 + 1.08863 w_k1_FDX_s011
                                 + 0.69803 w_k1_IBM_s011
                                 + 1.19337 w_k1_INTC_s011
                                 + 0.852954 w_k1_KO_s011
                                 + 0.528797 w_k1_MAT_s011
                                 + 1.04023 w_k1_MSFT_s011
                                 + 0.934702 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.945392 w_k1_ADM_s011
                                 + 0.832384 w_k1_AXP_s011
                                 + 0.756987 w_k1_DIS_s011
                                 + 0.848945 w_k1_FDX_s011
                                 + 0.833408 w_k1_IBM_s011
                                 + 0.849966 w_k1_INTC_s011
                                 + 0.793963 w_k1_KO_s011
                                 + 0.681979 w_k1_MAT_s011
                                 + 0.714267 w_k1_MSFT_s011
                                 + 0.816238 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.07593 w_k1_ADM_s011
                                 + 0.959897 w_k1_AXP_s011
                                 + 1.10316 w_k1_DIS_s011
                                 + 1.01036 w_k1_FDX_s011
                                 + 1.33426 w_k1_IBM_s011
                                 + 1.1394 w_k1_INTC_s011 + 1.15307 w_k1_KO_s011
                                 + 1.35965 w_k1_MAT_s011
                                 + 0.963806 w_k1_MSFT_s011
                                 + 1.09916 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.05958 w_k1_ADM_s011
                                 + 1.42179 w_k1_AXP_s011
                                 + 1.49299 w_k1_DIS_s011
                                 + 1.30183 w_k1_FDX_s011 + 1.346 w_k1_IBM_s011
                                 + 1.31699 w_k1_INTC_s011
                                 + 1.42853 w_k1_KO_s011 + 2.11606 w_k1_MAT_s011
                                 + 1.07561 w_k1_MSFT_s011
                                 + 1.20099 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.01112 w_k1_ADM_s012
                                 + 1.09888 w_k1_AXP_s012
                                 + 0.919895 w_k1_DIS_s012
                                 + 0.846034 w_k1_FDX_s012
                                 + 1.62311 w_k1_IBM_s012
                                 + 1.06597 w_k1_INTC_s012
                                 + 1.27612 w_k1_KO_s012 + 2.04967 w_k1_MAT_s012
                                 + 1.08737 w_k1_MSFT_s012
                                 + 1.26456 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.32102 w_k1_ADM_s012
                                 + 1.5102 w_k1_AXP_s012 + 1.83356 w_k1_DIS_s012
                                 + 1.52407 w_k1_FDX_s012
                                 + 1.05241 w_k1_IBM_s012
                                 + 1.28151 w_k1_INTC_s012
                                 + 1.39358 w_k1_KO_s012 + 1.47931 w_k1_MAT_s012
                                 + 1.60295 w_k1_MSFT_s012
                                 + 1.25753 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.01147 w_k1_ADM_s012
                                 + 0.895366 w_k1_AXP_s012
                                 + 0.705175 w_k1_DIS_s012
                                 + 0.827733 w_k1_FDX_s012
                                 + 0.623342 w_k1_IBM_s012
                                 + 0.821839 w_k1_INTC_s012
                                 + 0.693305 w_k1_KO_s012
                                 + 0.834082 w_k1_MAT_s012
                                 + 0.89378 w_k1_MSFT_s012
                                 + 0.972532 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.909758 w_k1_ADM_s012
                                 + 1.08891 w_k1_AXP_s012
                                 + 1.32204 w_k1_DIS_s012
                                 + 1.01714 w_k1_FDX_s012
                                 + 1.09046 w_k1_IBM_s012
                                 + 1.14337 w_k1_INTC_s012
                                 + 1.05503 w_k1_KO_s012 + 1.47369 w_k1_MAT_s012
                                 + 1.21414 w_k1_MSFT_s012
                                 + 1.19648 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.22317 w_k1_ADM_s012
                                 + 1.44435 w_k1_AXP_s012
                                 + 1.44808 w_k1_DIS_s012
                                 + 1.14024 w_k1_FDX_s012
                                 + 1.65947 w_k1_IBM_s012
                                 + 1.22308 w_k1_INTC_s012
                                 + 1.64862 w_k1_KO_s012 + 1.8455 w_k1_MAT_s012
                                 + 1.42035 w_k1_MSFT_s012
                                 + 1.33106 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.06219 w_k1_ADM_s012
                                 + 1.22354 w_k1_AXP_s012
                                 + 1.09212 w_k1_DIS_s012
                                 + 1.11332 w_k1_FDX_s012
                                 + 1.02386 w_k1_IBM_s012
                                 + 1.22991 w_k1_INTC_s012
                                 + 1.00954 w_k1_KO_s012 + 1.37816 w_k1_MAT_s012
                                 + 1.12243 w_k1_MSFT_s012
                                 + 1.12479 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.00891 w_k1_ADM_s012
                                 + 0.714395 w_k1_AXP_s012
                                 + 0.924178 w_k1_DIS_s012
                                 + 0.886457 w_k1_FDX_s012
                                 + 0.765686 w_k1_IBM_s012
                                 + 0.978591 w_k1_INTC_s012
                                 + 0.778482 w_k1_KO_s012
                                 + 0.599756 w_k1_MAT_s012
                                 + 1.00546 w_k1_MSFT_s012
                                 + 0.990629 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.974049 w_k1_ADM_s012
                                 + 0.970494 w_k1_AXP_s012
                                 + 1.07817 w_k1_DIS_s012
                                 + 0.885383 w_k1_FDX_s012
                                 + 1.33376 w_k1_IBM_s012
                                 + 1.07623 w_k1_INTC_s012
                                 + 1.02113 w_k1_KO_s012 + 1.40756 w_k1_MAT_s012
                                 + 0.968926 w_k1_MSFT_s012
                                 + 0.959654 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.2459 w_k1_ADM_s013
                                 + 1.56514 w_k1_AXP_s013
                                 + 1.39282 w_k1_DIS_s013
                                 + 1.39342 w_k1_FDX_s013
                                 + 1.32259 w_k1_IBM_s013
                                 + 1.27447 w_k1_INTC_s013
                                 + 1.13493 w_k1_KO_s013 + 1.84184 w_k1_MAT_s013
                                 + 1.11232 w_k1_MSFT_s013
                                 + 1.14697 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.990908 w_k1_ADM_s013
                                 + 1.3307 w_k1_AXP_s013 + 1.13474 w_k1_DIS_s013
                                 + 1.07156 w_k1_FDX_s013
                                 + 1.30409 w_k1_IBM_s013
                                 + 1.35873 w_k1_INTC_s013
                                 + 1.04819 w_k1_KO_s013 + 1.16964 w_k1_MAT_s013
                                 + 1.12111 w_k1_MSFT_s013
                                 + 1.07429 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.10265 w_k1_ADM_s013
                                 + 0.737113 w_k1_AXP_s013
                                 + 0.752024 w_k1_DIS_s013
                                 + 0.872617 w_k1_FDX_s013
                                 + 0.740307 w_k1_IBM_s013
                                 + 0.911358 w_k1_INTC_s013
                                 + 0.66184 w_k1_KO_s013
                                 + 0.547855 w_k1_MAT_s013
                                 + 0.941039 w_k1_MSFT_s013
                                 + 0.894943 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.700846 w_k1_ADM_s013
                                 + 0.594392 w_k1_AXP_s013
                                 + 0.611862 w_k1_DIS_s013
                                 + 0.650259 w_k1_FDX_s013
                                 + 0.754098 w_k1_IBM_s013
                                 + 0.846461 w_k1_INTC_s013
                                 + 0.669075 w_k1_KO_s013
                                 + 0.761406 w_k1_MAT_s013
                                 + 0.873928 w_k1_MSFT_s013
                                 + 0.821585 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.970163 w_k1_ADM_s013
                                 + 0.852831 w_k1_AXP_s013
                                 + 1.00606 w_k1_DIS_s013
                                 + 0.930191 w_k1_FDX_s013
                                 + 1.2658 w_k1_IBM_s013
                                 + 0.891382 w_k1_INTC_s013
                                 + 0.973151 w_k1_KO_s013
                                 + 1.16383 w_k1_MAT_s013
                                 + 0.939937 w_k1_MSFT_s013
                                 + 0.865364 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.704593 w_k1_ADM_s013
                                 + 0.931029 w_k1_AXP_s013
                                 + 0.953783 w_k1_DIS_s013
                                 + 0.942266 w_k1_FDX_s013
                                 + 0.943313 w_k1_IBM_s013
                                 + 0.926712 w_k1_INTC_s013
                                 + 0.766149 w_k1_KO_s013
                                 + 0.874848 w_k1_MAT_s013
                                 + 0.885807 w_k1_MSFT_s013
                                 + 0.8463 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.20951 w_k1_ADM_s013
                                 + 1.34785 w_k1_AXP_s013
                                 + 1.08215 w_k1_DIS_s013
                                 + 1.10313 w_k1_FDX_s013 + 1.0368 w_k1_IBM_s013
                                 + 1.2287 w_k1_INTC_s013
                                 + 0.920483 w_k1_KO_s013
                                 + 0.972696 w_k1_MAT_s013
                                 + 1.21566 w_k1_MSFT_s013
                                 + 0.990195 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.29913 w_k1_ADM_s013
                                 + 1.10971 w_k1_AXP_s013
                                 + 1.11275 w_k1_DIS_s013
                                 + 1.16591 w_k1_FDX_s013
                                 + 0.831642 w_k1_IBM_s013
                                 + 0.914263 w_k1_INTC_s013
                                 + 0.835878 w_k1_KO_s013
                                 + 0.810388 w_k1_MAT_s013
                                 + 1.09124 w_k1_MSFT_s013
                                 + 0.97653 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.661913 w_k1_ADM_s014
                                 + 0.577561 w_k1_AXP_s014
                                 + 0.642494 w_k1_DIS_s014
                                 + 0.700713 w_k1_FDX_s014
                                 + 0.783988 w_k1_IBM_s014
                                 + 0.587118 w_k1_INTC_s014
                                 + 0.712929 w_k1_KO_s014
                                 + 0.723698 w_k1_MAT_s014
                                 + 0.67445 w_k1_MSFT_s014
                                 + 0.798358 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.25745 w_k1_ADM_s014
                                 + 1.52476 w_k1_AXP_s014
                                 + 1.54546 w_k1_DIS_s014
                                 + 1.29244 w_k1_FDX_s014
                                 + 1.79518 w_k1_IBM_s014
                                 + 1.32935 w_k1_INTC_s014
                                 + 1.28444 w_k1_KO_s014 + 1.48493 w_k1_MAT_s014
                                 + 1.16743 w_k1_MSFT_s014
                                 + 1.08826 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.863571 w_k1_ADM_s014
                                 + 0.842293 w_k1_AXP_s014
                                 + 0.646939 w_k1_DIS_s014
                                 + 0.759629 w_k1_FDX_s014
                                 + 1.11873 w_k1_IBM_s014
                                 + 0.77742 w_k1_INTC_s014
                                 + 0.761152 w_k1_KO_s014
                                 + 0.807073 w_k1_MAT_s014
                                 + 0.727773 w_k1_MSFT_s014
                                 + 0.857026 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.09536 w_k1_ADM_s014
                                 + 1.02525 w_k1_AXP_s014
                                 + 0.91758 w_k1_DIS_s014
                                 + 1.01849 w_k1_FDX_s014
                                 + 0.822718 w_k1_IBM_s014
                                 + 0.92518 w_k1_INTC_s014
                                 + 0.765589 w_k1_KO_s014
                                 + 0.711225 w_k1_MAT_s014
                                 + 0.966032 w_k1_MSFT_s014
                                 + 1.02026 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.08701 w_k1_ADM_s014
                                 + 1.45634 w_k1_AXP_s014
                                 + 1.41436 w_k1_DIS_s014
                                 + 1.32015 w_k1_FDX_s014
                                 + 1.03473 w_k1_IBM_s014
                                 + 1.23513 w_k1_INTC_s014
                                 + 0.928139 w_k1_KO_s014
                                 + 1.16657 w_k1_MAT_s014
                                 + 1.32604 w_k1_MSFT_s014
                                 + 0.996785 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.11733 w_k1_ADM_s014
                                 + 0.912999 w_k1_AXP_s014
                                 + 0.944855 w_k1_DIS_s014
                                 + 0.965678 w_k1_FDX_s014
                                 + 1.03073 w_k1_IBM_s014
                                 + 0.945952 w_k1_INTC_s014
                                 + 0.994701 w_k1_KO_s014
                                 + 1.22005 w_k1_MAT_s014
                                 + 0.974016 w_k1_MSFT_s014
                                 + 1.01367 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.971836 w_k1_ADM_s014
                                 + 0.757597 w_k1_AXP_s014
                                 + 0.760984 w_k1_DIS_s014
                                 + 0.861318 w_k1_FDX_s014
                                 + 0.617203 w_k1_IBM_s014
                                 + 0.842479 w_k1_INTC_s014
                                 + 0.731977 w_k1_KO_s014
                                 + 0.634153 w_k1_MAT_s014
                                 + 0.836565 w_k1_MSFT_s014
                                 + 0.782697 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.760385 w_k1_ADM_s014
                                 + 1.10449 w_k1_AXP_s014
                                 + 1.07273 w_k1_DIS_s014
                                 + 1.04225 w_k1_FDX_s014
                                 + 1.40971 w_k1_IBM_s014
                                 + 1.09299 w_k1_INTC_s014 + 1.0155 w_k1_KO_s014
                                 + 1.33031 w_k1_MAT_s014
                                 + 0.958827 w_k1_MSFT_s014
                                 + 1.04475 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.797812 w_k1_ADM_s015
                                 + 0.766209 w_k1_AXP_s015
                                 + 0.709581 w_k1_DIS_s015
                                 + 0.748245 w_k1_FDX_s015
                                 + 1.10798 w_k1_IBM_s015
                                 + 1.01045 w_k1_INTC_s015
                                 + 1.01566 w_k1_KO_s015 + 1.19997 w_k1_MAT_s015
                                 + 0.900939 w_k1_MSFT_s015
                                 + 0.906241 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.32635 w_k1_ADM_s015
                                 + 1.14844 w_k1_AXP_s015
                                 + 1.78748 w_k1_DIS_s015
                                 + 1.48333 w_k1_FDX_s015 + 1.0685 w_k1_IBM_s015
                                 + 1.44851 w_k1_INTC_s015
                                 + 1.38809 w_k1_KO_s015
                                 + 0.938715 w_k1_MAT_s015
                                 + 1.58465 w_k1_MSFT_s015
                                 + 1.22253 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.13909 w_k1_ADM_s015
                                 + 1.42042 w_k1_AXP_s015
                                 + 1.31504 w_k1_DIS_s015
                                 + 1.24435 w_k1_FDX_s015
                                 + 1.35303 w_k1_IBM_s015
                                 + 1.21254 w_k1_INTC_s015
                                 + 1.22701 w_k1_KO_s015 + 1.66288 w_k1_MAT_s015
                                 + 1.38895 w_k1_MSFT_s015
                                 + 1.13074 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.1625 w_k1_ADM_s015
                                 + 0.823393 w_k1_AXP_s015
                                 + 1.52902 w_k1_DIS_s015 + 1.1279 w_k1_FDX_s015
                                 + 1.10322 w_k1_IBM_s015
                                 + 1.14699 w_k1_INTC_s015
                                 + 1.11341 w_k1_KO_s015
                                 + 0.716893 w_k1_MAT_s015
                                 + 1.38553 w_k1_MSFT_s015
                                 + 1.04842 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.04756 w_k1_ADM_s015
                                 + 1.24661 w_k1_AXP_s015
                                 + 1.18313 w_k1_DIS_s015
                                 + 0.960123 w_k1_FDX_s015
                                 + 1.53915 w_k1_IBM_s015
                                 + 1.05171 w_k1_INTC_s015
                                 + 1.18272 w_k1_KO_s015 + 1.80893 w_k1_MAT_s015
                                 + 1.20454 w_k1_MSFT_s015
                                 + 1.07111 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.05093 w_k1_ADM_s015
                                 + 1.04985 w_k1_AXP_s015 + 1.1379 w_k1_DIS_s015
                                 + 0.981224 w_k1_FDX_s015
                                 + 1.04288 w_k1_IBM_s015
                                 + 1.09742 w_k1_INTC_s015
                                 + 1.00992 w_k1_KO_s015
                                 + 0.968454 w_k1_MAT_s015
                                 + 1.03913 w_k1_MSFT_s015
                                 + 0.945714 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.25202 w_k1_ADM_s015
                                 + 1.27507 w_k1_AXP_s015
                                 + 1.03836 w_k1_DIS_s015
                                 + 1.15405 w_k1_FDX_s015
                                 + 1.06119 w_k1_IBM_s015
                                 + 1.05269 w_k1_INTC_s015
                                 + 1.19692 w_k1_KO_s015 + 1.31718 w_k1_MAT_s015
                                 + 1.13897 w_k1_MSFT_s015
                                 + 1.06095 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.895898 w_k1_ADM_s015
                                 + 0.784224 w_k1_AXP_s015
                                 + 0.79298 w_k1_DIS_s015
                                 + 0.940223 w_k1_FDX_s015
                                 + 0.66215 w_k1_IBM_s015 + 1.11 w_k1_INTC_s015
                                 + 0.832693 w_k1_KO_s015
                                 + 0.683662 w_k1_MAT_s015
                                 + 1.07933 w_k1_MSFT_s015
                                 + 0.887282 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.782726 w_k1_ADM_s016
                                 + 0.891979 w_k1_AXP_s016
                                 + 0.732687 w_k1_DIS_s016
                                 + 0.887461 w_k1_FDX_s016
                                 + 0.612823 w_k1_IBM_s016
                                 + 0.814047 w_k1_INTC_s016
                                 + 0.682649 w_k1_KO_s016
                                 + 0.523215 w_k1_MAT_s016
                                 + 0.897156 w_k1_MSFT_s016
                                 + 0.81822 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.00155 w_k1_ADM_s016
                                 + 1.21118 w_k1_AXP_s016
                                 + 1.03567 w_k1_DIS_s016
                                 + 1.00728 w_k1_FDX_s016
                                 + 1.35714 w_k1_IBM_s016
                                 + 0.970516 w_k1_INTC_s016
                                 + 1.17092 w_k1_KO_s016 + 1.4833 w_k1_MAT_s016
                                 + 1.03869 w_k1_MSFT_s016
                                 + 1.06902 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.21393 w_k1_ADM_s016
                                 + 1.06021 w_k1_AXP_s016 + 1.1964 w_k1_DIS_s016
                                 + 1.23284 w_k1_FDX_s016
                                 + 0.78534 w_k1_IBM_s016
                                 + 1.04886 w_k1_INTC_s016
                                 + 1.03891 w_k1_KO_s016
                                 + 0.747784 w_k1_MAT_s016
                                 + 1.22227 w_k1_MSFT_s016
                                 + 1.03224 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.862408 w_k1_ADM_s016
                                 + 0.841212 w_k1_AXP_s016
                                 + 0.587831 w_k1_DIS_s016
                                 + 0.812438 w_k1_FDX_s016
                                 + 0.875522 w_k1_IBM_s016
                                 + 0.705676 w_k1_INTC_s016
                                 + 0.680711 w_k1_KO_s016
                                 + 0.806759 w_k1_MAT_s016
                                 + 0.771289 w_k1_MSFT_s016
                                 + 0.813003 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.655043 w_k1_ADM_s016
                                 + 0.795453 w_k1_AXP_s016
                                 + 0.834251 w_k1_DIS_s016
                                 + 0.798435 w_k1_FDX_s016
                                 + 0.760196 w_k1_IBM_s016
                                 + 0.83852 w_k1_INTC_s016
                                 + 0.815037 w_k1_KO_s016
                                 + 0.811174 w_k1_MAT_s016
                                 + 0.894175 w_k1_MSFT_s016
                                 + 0.907252 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.961977 w_k1_ADM_s016
                                 + 0.952111 w_k1_AXP_s016
                                 + 1.10512 w_k1_DIS_s016
                                 + 1.05805 w_k1_FDX_s016
                                 + 0.915838 w_k1_IBM_s016
                                 + 0.932937 w_k1_INTC_s016
                                 + 0.985992 w_k1_KO_s016
                                 + 0.664282 w_k1_MAT_s016
                                 + 0.991112 w_k1_MSFT_s016
                                 + 0.917272 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.08135 w_k1_ADM_s016
                                 + 1.47523 w_k1_AXP_s016
                                 + 1.64745 w_k1_DIS_s016
                                 + 1.40058 w_k1_FDX_s016
                                 + 1.26447 w_k1_IBM_s016
                                 + 0.924666 w_k1_INTC_s016
                                 + 1.28515 w_k1_KO_s016 + 1.55509 w_k1_MAT_s016
                                 + 1.28382 w_k1_MSFT_s016
                                 + 1.16378 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.836002 w_k1_ADM_s016
                                 + 0.806586 w_k1_AXP_s016
                                 + 0.664588 w_k1_DIS_s016
                                 + 0.752113 w_k1_FDX_s016
                                 + 0.999431 w_k1_IBM_s016
                                 + 0.828853 w_k1_INTC_s016
                                 + 0.953606 w_k1_KO_s016
                                 + 0.837589 w_k1_MAT_s016
                                 + 0.868854 w_k1_MSFT_s016
                                 + 0.897631 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.97344 w_k1_ADM_s017
                                 + 1.3313 w_k1_AXP_s017 + 1.36179 w_k1_DIS_s017
                                 + 1.23518 w_k1_FDX_s017
                                 + 1.50033 w_k1_IBM_s017
                                 + 1.11144 w_k1_INTC_s017
                                 + 1.21713 w_k1_KO_s017 + 1.48208 w_k1_MAT_s017
                                 + 1.05829 w_k1_MSFT_s017 + 1.0386 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.08916 w_k1_ADM_s017
                                 + 1.12484 w_k1_AXP_s017
                                 + 1.49751 w_k1_DIS_s017
                                 + 1.24851 w_k1_FDX_s017 + 1.1208 w_k1_IBM_s017
                                 + 1.05186 w_k1_INTC_s017 + 1.1336 w_k1_KO_s017
                                 + 1.24267 w_k1_MAT_s017
                                 + 1.14001 w_k1_MSFT_s017
                                 + 1.03398 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.838736 w_k1_ADM_s017
                                 + 0.893193 w_k1_AXP_s017
                                 + 0.875878 w_k1_DIS_s017
                                 + 0.937851 w_k1_FDX_s017
                                 + 0.773095 w_k1_IBM_s017
                                 + 0.891528 w_k1_INTC_s017
                                 + 0.799125 w_k1_KO_s017 + 0.7244 w_k1_MAT_s017
                                 + 0.839606 w_k1_MSFT_s017
                                 + 0.819051 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.11011 w_k1_ADM_s017
                                 + 1.07865 w_k1_AXP_s017 + 1.0148 w_k1_DIS_s017
                                 + 1.15028 w_k1_FDX_s017
                                 + 0.736095 w_k1_IBM_s017
                                 + 0.933472 w_k1_INTC_s017
                                 + 1.01957 w_k1_KO_s017
                                 + 0.853406 w_k1_MAT_s017
                                 + 1.07402 w_k1_MSFT_s017
                                 + 0.984738 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.654154 w_k1_ADM_s017
                                 + 0.609818 w_k1_AXP_s017
                                 + 0.641685 w_k1_DIS_s017
                                 + 0.774712 w_k1_FDX_s017
                                 + 0.967942 w_k1_IBM_s017
                                 + 0.708684 w_k1_INTC_s017
                                 + 0.753628 w_k1_KO_s017
                                 + 0.956288 w_k1_MAT_s017
                                 + 0.739175 w_k1_MSFT_s017
                                 + 0.804926 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.893387 w_k1_ADM_s017
                                 + 0.65098 w_k1_AXP_s017
                                 + 0.573972 w_k1_DIS_s017
                                 + 0.761817 w_k1_FDX_s017
                                 + 0.788428 w_k1_IBM_s017
                                 + 0.655677 w_k1_INTC_s017
                                 + 0.727247 w_k1_KO_s017
                                 + 0.673758 w_k1_MAT_s017
                                 + 0.702479 w_k1_MSFT_s017
                                 + 0.761125 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.779496 w_k1_ADM_s017
                                 + 0.985029 w_k1_AXP_s017
                                 + 0.862948 w_k1_DIS_s017
                                 + 0.914824 w_k1_FDX_s017
                                 + 1.30729 w_k1_IBM_s017
                                 + 0.860724 w_k1_INTC_s017
                                 + 0.945494 w_k1_KO_s017
                                 + 1.15579 w_k1_MAT_s017
                                 + 0.869427 w_k1_MSFT_s017
                                 + 0.895479 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.00344 w_k1_ADM_s017
                                 + 1.03202 w_k1_AXP_s017
                                 + 1.02147 w_k1_DIS_s017
                                 + 1.03945 w_k1_FDX_s017
                                 + 0.96924 w_k1_IBM_s017
                                 + 0.847611 w_k1_INTC_s017
                                 + 0.979037 w_k1_KO_s017
                                 + 1.19278 w_k1_MAT_s017
                                 + 1.02588 w_k1_MSFT_s017
                                 + 1.07936 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.07475 w_k1_ADM_s100
                                 + 1.24248 w_k1_AXP_s100
                                 + 1.15335 w_k1_DIS_s100
                                 + 1.02666 w_k1_FDX_s100
                                 + 1.41652 w_k1_IBM_s100
                                 + 1.05082 w_k1_INTC_s100
                                 + 1.17941 w_k1_KO_s100 + 1.56993 w_k1_MAT_s100
                                 + 1.06511 w_k1_MSFT_s100
                                 + 0.978419 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.716784 w_k1_ADM_s100
                                 + 0.962141 w_k1_AXP_s100
                                 + 0.95323 w_k1_DIS_s100
                                 + 0.975267 w_k1_FDX_s100
                                 + 1.04956 w_k1_IBM_s100
                                 + 1.05127 w_k1_INTC_s100
                                 + 0.962501 w_k1_KO_s100
                                 + 1.11799 w_k1_MAT_s100
                                 + 0.988251 w_k1_MSFT_s100
                                 + 0.879449 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.41037 w_k1_ADM_s100
                                 + 1.08179 w_k1_AXP_s100 + 1.14 w_k1_DIS_s100
                                 + 1.22944 w_k1_FDX_s100
                                 + 0.88406 w_k1_IBM_s100
                                 + 1.00787 w_k1_INTC_s100
                                 + 1.02729 w_k1_KO_s100
                                 + 0.783653 w_k1_MAT_s100
                                 + 1.08794 w_k1_MSFT_s100
                                 + 0.960042 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.683684 w_k1_ADM_s100
                                 + 0.743404 w_k1_AXP_s100
                                 + 0.604989 w_k1_DIS_s100
                                 + 0.745855 w_k1_FDX_s100
                                 + 0.720088 w_k1_IBM_s100
                                 + 0.822964 w_k1_INTC_s100
                                 + 0.795045 w_k1_KO_s100
                                 + 0.944285 w_k1_MAT_s100
                                 + 0.847738 w_k1_MSFT_s100
                                 + 0.765458 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.970382 w_k1_ADM_s100
                                 + 0.895695 w_k1_AXP_s100
                                 + 0.849437 w_k1_DIS_s100
                                 + 0.98029 w_k1_FDX_s100
                                 + 0.934048 w_k1_IBM_s100
                                 + 0.809613 w_k1_INTC_s100
                                 + 0.881916 w_k1_KO_s100
                                 + 0.999177 w_k1_MAT_s100
                                 + 0.922653 w_k1_MSFT_s100
                                 + 0.86564 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.926505 w_k1_ADM_s100
                                 + 0.762291 w_k1_AXP_s100
                                 + 0.818042 w_k1_DIS_s100
                                 + 0.789795 w_k1_FDX_s100
                                 + 1.20887 w_k1_IBM_s100
                                 + 0.724922 w_k1_INTC_s100
                                 + 0.977705 w_k1_KO_s100 + 1.2016 w_k1_MAT_s100
                                 + 0.879531 w_k1_MSFT_s100
                                 + 0.812854 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.30075 w_k1_ADM_s100
                                 + 1.27376 w_k1_AXP_s100
                                 + 1.65953 w_k1_DIS_s100 + 1.282 w_k1_FDX_s100
                                 + 1.47925 w_k1_IBM_s100
                                 + 1.01678 w_k1_INTC_s100
                                 + 1.49132 w_k1_KO_s100 + 1.96029 w_k1_MAT_s100
                                 + 1.22041 w_k1_MSFT_s100 + 1.1426 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.08959 w_k1_ADM_s100
                                 + 0.802338 w_k1_AXP_s100
                                 + 0.794431 w_k1_DIS_s100
                                 + 0.969619 w_k1_FDX_s100
                                 + 0.682047 w_k1_IBM_s100
                                 + 0.858651 w_k1_INTC_s100
                                 + 0.75552 w_k1_KO_s100
                                 + 0.689926 w_k1_MAT_s100
                                 + 0.960446 w_k1_MSFT_s100
                                 + 0.810968 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.889472 w_k1_ADM_s101
                                 + 0.818238 w_k1_AXP_s101 + 1.032 w_k1_DIS_s101
                                 + 0.938615 w_k1_FDX_s101
                                 + 0.93176 w_k1_IBM_s101
                                 + 1.00536 w_k1_INTC_s101
                                 + 1.19445 w_k1_KO_s101
                                 + 0.721791 w_k1_MAT_s101
                                 + 0.988185 w_k1_MSFT_s101
                                 + 0.906831 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.893565 w_k1_ADM_s101
                                 + 0.704826 w_k1_AXP_s101
                                 + 0.606127 w_k1_DIS_s101
                                 + 0.780939 w_k1_FDX_s101
                                 + 0.719626 w_k1_IBM_s101
                                 + 0.91862 w_k1_INTC_s101
                                 + 0.790412 w_k1_KO_s101
                                 + 0.632235 w_k1_MAT_s101
                                 + 0.768696 w_k1_MSFT_s101
                                 + 0.751224 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.999337 w_k1_ADM_s101
                                 + 1.15495 w_k1_AXP_s101
                                 + 0.70894 w_k1_DIS_s101
                                 + 0.841982 w_k1_FDX_s101
                                 + 1.00391 w_k1_IBM_s101
                                 + 0.742006 w_k1_INTC_s101
                                 + 0.905001 w_k1_KO_s101
                                 + 1.12585 w_k1_MAT_s101
                                 + 0.699958 w_k1_MSFT_s101
                                 + 0.839341 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.810256 w_k1_ADM_s101
                                 + 0.763557 w_k1_AXP_s101
                                 + 0.737075 w_k1_DIS_s101
                                 + 0.833124 w_k1_FDX_s101
                                 + 0.89972 w_k1_IBM_s101
                                 + 0.855111 w_k1_INTC_s101
                                 + 0.910874 w_k1_KO_s101
                                 + 0.787897 w_k1_MAT_s101
                                 + 0.76378 w_k1_MSFT_s101
                                 + 0.851007 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.14804 w_k1_ADM_s101
                                 + 1.17628 w_k1_AXP_s101
                                 + 1.02574 w_k1_DIS_s101
                                 + 1.05152 w_k1_FDX_s101
                                 + 1.09316 w_k1_IBM_s101
                                 + 1.06316 w_k1_INTC_s101
                                 + 1.15872 w_k1_KO_s101 + 1.08102 w_k1_MAT_s101
                                 + 1.05435 w_k1_MSFT_s101
                                 + 1.00616 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.13055 w_k1_ADM_s101
                                 + 1.02864 w_k1_AXP_s101
                                 + 0.960706 w_k1_DIS_s101
                                 + 1.06999 w_k1_FDX_s101
                                 + 0.747989 w_k1_IBM_s101
                                 + 1.0709 w_k1_INTC_s101
                                 + 0.996117 w_k1_KO_s101
                                 + 0.703807 w_k1_MAT_s101
                                 + 0.973212 w_k1_MSFT_s101
                                 + 0.933952 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.26045 w_k1_ADM_s101
                                 + 1.43442 w_k1_AXP_s101
                                 + 1.58012 w_k1_DIS_s101
                                 + 1.40687 w_k1_FDX_s101
                                 + 1.11239 w_k1_IBM_s101
                                 + 1.18871 w_k1_INTC_s101 + 1.3235 w_k1_KO_s101
                                 + 1.20574 w_k1_MAT_s101
                                 + 1.26004 w_k1_MSFT_s101
                                 + 1.15729 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.995241 w_k1_ADM_s101
                                 + 1.30848 w_k1_AXP_s101
                                 + 1.36354 w_k1_DIS_s101
                                 + 1.12475 w_k1_FDX_s101
                                 + 1.53161 w_k1_IBM_s101
                                 + 1.12712 w_k1_INTC_s101
                                 + 1.45134 w_k1_KO_s101 + 1.79381 w_k1_MAT_s101
                                 + 1.10396 w_k1_MSFT_s101
                                 + 0.992854 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.724861 w_k1_ADM_s102
                                 + 0.842382 w_k1_AXP_s102
                                 + 0.880394 w_k1_DIS_s102
                                 + 0.982231 w_k1_FDX_s102
                                 + 0.606934 w_k1_IBM_s102
                                 + 0.936803 w_k1_INTC_s102
                                 + 0.709949 w_k1_KO_s102
                                 + 0.669297 w_k1_MAT_s102
                                 + 0.883466 w_k1_MSFT_s102
                                 + 0.780557 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.826326 w_k1_ADM_s102
                                 + 0.886957 w_k1_AXP_s102
                                 + 0.735079 w_k1_DIS_s102
                                 + 0.804834 w_k1_FDX_s102
                                 + 1.0211 w_k1_IBM_s102
                                 + 0.88993 w_k1_INTC_s102
                                 + 0.872468 w_k1_KO_s102
                                 + 0.978762 w_k1_MAT_s102
                                 + 0.799431 w_k1_MSFT_s102
                                 + 0.748523 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.864994 w_k1_ADM_s102
                                 + 0.944509 w_k1_AXP_s102
                                 + 1.09776 w_k1_DIS_s102
                                 + 1.01618 w_k1_FDX_s102
                                 + 0.789796 w_k1_IBM_s102
                                 + 0.8718 w_k1_INTC_s102
                                 + 0.942432 w_k1_KO_s102 + 0.7087 w_k1_MAT_s102
                                 + 0.92418 w_k1_MSFT_s102
                                 + 0.894682 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.05384 w_k1_ADM_s102
                                 + 0.975025 w_k1_AXP_s102
                                 + 1.32989 w_k1_DIS_s102
                                 + 1.14875 w_k1_FDX_s102
                                 + 1.06035 w_k1_IBM_s102
                                 + 1.14009 w_k1_INTC_s102
                                 + 1.16474 w_k1_KO_s102 + 0.81391 w_k1_MAT_s102
                                 + 1.23045 w_k1_MSFT_s102
                                 + 0.985089 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.21561 w_k1_ADM_s102
                                 + 1.64452 w_k1_AXP_s102
                                 + 1.83311 w_k1_DIS_s102
                                 + 1.59784 w_k1_FDX_s102
                                 + 1.11148 w_k1_IBM_s102
                                 + 1.52583 w_k1_INTC_s102
                                 + 1.27562 w_k1_KO_s102
                                 + 0.970666 w_k1_MAT_s102
                                 + 1.46629 w_k1_MSFT_s102
                                 + 1.11995 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.880645 w_k1_ADM_s102
                                 + 1.44283 w_k1_AXP_s102
                                 + 1.25772 w_k1_DIS_s102
                                 + 1.08437 w_k1_FDX_s102
                                 + 1.99243 w_k1_IBM_s102
                                 + 1.19861 w_k1_INTC_s102
                                 + 1.39444 w_k1_KO_s102 + 2.09854 w_k1_MAT_s102
                                 + 1.16846 w_k1_MSFT_s102
                                 + 0.984398 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.936806 w_k1_ADM_s102
                                 + 1.10161 w_k1_AXP_s102
                                 + 1.23901 w_k1_DIS_s102
                                 + 1.06171 w_k1_FDX_s102
                                 + 1.40455 w_k1_IBM_s102
                                 + 1.06756 w_k1_INTC_s102
                                 + 1.20403 w_k1_KO_s102 + 1.07615 w_k1_MAT_s102
                                 + 0.997164 w_k1_MSFT_s102
                                 + 0.912421 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.875791 w_k1_ADM_s102
                                 + 1.00896 w_k1_AXP_s102
                                 + 0.934828 w_k1_DIS_s102
                                 + 0.795714 w_k1_FDX_s102
                                 + 1.59459 w_k1_IBM_s102
                                 + 0.846492 w_k1_INTC_s102
                                 + 1.12101 w_k1_KO_s102 + 1.35216 w_k1_MAT_s102
                                 + 0.899049 w_k1_MSFT_s102
                                 + 0.932471 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.02951 w_k1_ADM_s103
                                 + 1.03975 w_k1_AXP_s103
                                 + 0.849833 w_k1_DIS_s103
                                 + 0.881924 w_k1_FDX_s103
                                 + 1.51117 w_k1_IBM_s103
                                 + 0.885612 w_k1_INTC_s103
                                 + 0.969037 w_k1_KO_s103
                                 + 1.51442 w_k1_MAT_s103
                                 + 0.929865 w_k1_MSFT_s103
                                 + 1.11915 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.14658 w_k1_ADM_s103
                                 + 1.04146 w_k1_AXP_s103
                                 + 1.06325 w_k1_DIS_s103
                                 + 1.17985 w_k1_FDX_s103
                                 + 0.746586 w_k1_IBM_s103
                                 + 1.07686 w_k1_INTC_s103
                                 + 0.817547 w_k1_KO_s103
                                 + 0.736048 w_k1_MAT_s103
                                 + 1.04845 w_k1_MSFT_s103
                                 + 1.06172 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.00445 w_k1_ADM_s103
                                 + 0.788736 w_k1_AXP_s103
                                 + 0.740422 w_k1_DIS_s103
                                 + 0.998229 w_k1_FDX_s103
                                 + 0.597949 w_k1_IBM_s103
                                 + 0.818947 w_k1_INTC_s103
                                 + 0.734798 w_k1_KO_s103
                                 + 0.612152 w_k1_MAT_s103
                                 + 0.86035 w_k1_MSFT_s103
                                 + 0.968632 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.40079 w_k1_ADM_s103
                                 + 1.63599 w_k1_AXP_s103
                                 + 1.58545 w_k1_DIS_s103
                                 + 1.27812 w_k1_FDX_s103
                                 + 1.68699 w_k1_IBM_s103
                                 + 1.4551 w_k1_INTC_s103 + 1.31059 w_k1_KO_s103
                                 + 2.03891 w_k1_MAT_s103
                                 + 1.21409 w_k1_MSFT_s103 + 1.313 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.14907 w_k1_ADM_s103
                                 + 1.32498 w_k1_AXP_s103
                                 + 1.62093 w_k1_DIS_s103 + 1.2648 w_k1_FDX_s103
                                 + 1.24329 w_k1_IBM_s103
                                 + 1.16492 w_k1_INTC_s103
                                 + 1.21255 w_k1_KO_s103 + 1.46665 w_k1_MAT_s103
                                 + 1.28602 w_k1_MSFT_s103
                                 + 1.30551 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.25577 w_k1_ADM_s103
                                 + 1.36477 w_k1_AXP_s103
                                 + 1.39788 w_k1_DIS_s103
                                 + 1.29543 w_k1_FDX_s103
                                 + 0.935288 w_k1_IBM_s103
                                 + 1.13804 w_k1_INTC_s103
                                 + 0.913539 w_k1_KO_s103
                                 + 1.14997 w_k1_MAT_s103
                                 + 1.25109 w_k1_MSFT_s103
                                 + 1.19954 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.877685 w_k1_ADM_s103
                                 + 0.648299 w_k1_AXP_s103
                                 + 0.526801 w_k1_DIS_s103
                                 + 0.718994 w_k1_FDX_s103
                                 + 0.740283 w_k1_IBM_s103
                                 + 0.898174 w_k1_INTC_s103
                                 + 0.639578 w_k1_KO_s103
                                 + 0.642466 w_k1_MAT_s103
                                 + 0.73297 w_k1_MSFT_s103
                                 + 0.76703 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.16266 w_k1_ADM_s103
                                 + 0.975971 w_k1_AXP_s103
                                 + 0.77468 w_k1_DIS_s103
                                 + 0.975374 w_k1_FDX_s103
                                 + 0.999512 w_k1_IBM_s103
                                 + 0.944534 w_k1_INTC_s103
                                 + 0.804173 w_k1_KO_s103
                                 + 1.15815 w_k1_MAT_s103
                                 + 0.91119 w_k1_MSFT_s103
                                 + 0.958099 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.14656 w_k1_ADM_s104
                                 + 1.09644 w_k1_AXP_s104
                                 + 0.986966 w_k1_DIS_s104
                                 + 0.944192 w_k1_FDX_s104
                                 + 1.14306 w_k1_IBM_s104
                                 + 0.866256 w_k1_INTC_s104
                                 + 1.11829 w_k1_KO_s104 + 1.19103 w_k1_MAT_s104
                                 + 1.01064 w_k1_MSFT_s104
                                 + 0.947654 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.26556 w_k1_ADM_s104
                                 + 1.25648 w_k1_AXP_s104
                                 + 1.53345 w_k1_DIS_s104
                                 + 1.31717 w_k1_FDX_s104
                                 + 1.00558 w_k1_IBM_s104
                                 + 0.998585 w_k1_INTC_s104
                                 + 1.27153 w_k1_KO_s104 + 1.37057 w_k1_MAT_s104
                                 + 1.20386 w_k1_MSFT_s104
                                 + 1.08874 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.969364 w_k1_ADM_s104
                                 + 1.03016 w_k1_AXP_s104
                                 + 1.04024 w_k1_DIS_s104
                                 + 1.09821 w_k1_FDX_s104
                                 + 0.741764 w_k1_IBM_s104
                                 + 0.944818 w_k1_INTC_s104
                                 + 0.913689 w_k1_KO_s104
                                 + 0.90121 w_k1_MAT_s104
                                 + 1.07291 w_k1_MSFT_s104
                                 + 0.966861 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.760578 w_k1_ADM_s104
                                 + 0.812407 w_k1_AXP_s104
                                 + 0.64717 w_k1_DIS_s104 + 0.7829 w_k1_FDX_s104
                                 + 0.673447 w_k1_IBM_s104
                                 + 0.786266 w_k1_INTC_s104
                                 + 0.651124 w_k1_KO_s104
                                 + 0.576395 w_k1_MAT_s104
                                 + 0.740636 w_k1_MSFT_s104
                                 + 0.653911 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.895145 w_k1_ADM_s104
                                 + 1.0483 w_k1_AXP_s104
                                 + 0.825034 w_k1_DIS_s104
                                 + 0.876431 w_k1_FDX_s104
                                 + 0.949838 w_k1_IBM_s104
                                 + 0.79368 w_k1_INTC_s104
                                 + 0.844303 w_k1_KO_s104
                                 + 1.04193 w_k1_MAT_s104
                                 + 0.927933 w_k1_MSFT_s104
                                 + 0.922531 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.0309 w_k1_ADM_s104
                                 + 1.45145 w_k1_AXP_s104
                                 + 1.36995 w_k1_DIS_s104
                                 + 1.02667 w_k1_FDX_s104
                                 + 1.87391 w_k1_IBM_s104
                                 + 0.967865 w_k1_INTC_s104
                                 + 1.49039 w_k1_KO_s104 + 1.98143 w_k1_MAT_s104
                                 + 1.07573 w_k1_MSFT_s104 + 1.1372 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.998732 w_k1_ADM_s104
                                 + 1.07835 w_k1_AXP_s104
                                 + 0.752898 w_k1_DIS_s104
                                 + 0.807833 w_k1_FDX_s104
                                 + 1.35209 w_k1_IBM_s104
                                 + 0.788409 w_k1_INTC_s104
                                 + 1.08956 w_k1_KO_s104 + 1.70781 w_k1_MAT_s104
                                 + 0.860765 w_k1_MSFT_s104
                                 + 0.908828 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.16302 w_k1_ADM_s104
                                 + 1.50217 w_k1_AXP_s104
                                 + 1.32541 w_k1_DIS_s104
                                 + 1.21843 w_k1_FDX_s104
                                 + 1.36606 w_k1_IBM_s104
                                 + 1.04585 w_k1_INTC_s104
                                 + 1.17885 w_k1_KO_s104 + 1.47011 w_k1_MAT_s104
                                 + 1.1215 w_k1_MSFT_s104 + 1.08017 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.18934 w_k1_ADM_s105
                                 + 0.711565 w_k1_AXP_s105
                                 + 0.7044 w_k1_DIS_s105
                                 + 0.953099 w_k1_FDX_s105
                                 + 0.500191 w_k1_IBM_s105
                                 + 0.862087 w_k1_INTC_s105
                                 + 0.615496 w_k1_KO_s105
                                 + 0.400608 w_k1_MAT_s105
                                 + 0.861945 w_k1_MSFT_s105
                                 + 0.88545 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.54572 w_k1_ADM_s105
                                 + 1.00816 w_k1_AXP_s105
                                 + 1.42995 w_k1_DIS_s105
                                 + 1.35947 w_k1_FDX_s105
                                 + 1.13212 w_k1_IBM_s105
                                 + 1.22016 w_k1_INTC_s105 + 1.2719 w_k1_KO_s105
                                 + 0.775723 w_k1_MAT_s105
                                 + 1.34368 w_k1_MSFT_s105
                                 + 1.20018 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.778563 w_k1_ADM_s105
                                 + 0.679123 w_k1_AXP_s105
                                 + 0.777483 w_k1_DIS_s105
                                 + 0.796025 w_k1_FDX_s105
                                 + 0.953316 w_k1_IBM_s105
                                 + 0.771865 w_k1_INTC_s105
                                 + 0.862903 w_k1_KO_s105
                                 + 0.825787 w_k1_MAT_s105
                                 + 0.828786 w_k1_MSFT_s105
                                 + 0.910792 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.30619 w_k1_ADM_s105
                                 + 0.976852 w_k1_AXP_s105
                                 + 1.05046 w_k1_DIS_s105
                                 + 1.10405 w_k1_FDX_s105
                                 + 0.792213 w_k1_IBM_s105
                                 + 0.967472 w_k1_INTC_s105
                                 + 0.813684 w_k1_KO_s105
                                 + 0.646513 w_k1_MAT_s105
                                 + 1.01562 w_k1_MSFT_s105
                                 + 0.906471 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.93351 w_k1_ADM_s105
                                 + 1.5053 w_k1_AXP_s105 + 1.31026 w_k1_DIS_s105
                                 + 1.36335 w_k1_FDX_s105
                                 + 1.47152 w_k1_IBM_s105
                                 + 1.06254 w_k1_INTC_s105
                                 + 1.56698 w_k1_KO_s105 + 1.51456 w_k1_MAT_s105
                                 + 1.09089 w_k1_MSFT_s105
                                 + 1.30335 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.0069 w_k1_ADM_s105
                                 + 0.551382 w_k1_AXP_s105
                                 + 0.642749 w_k1_DIS_s105
                                 + 0.836315 w_k1_FDX_s105
                                 + 0.649121 w_k1_IBM_s105
                                 + 0.833676 w_k1_INTC_s105
                                 + 0.686134 w_k1_KO_s105
                                 + 0.529978 w_k1_MAT_s105
                                 + 0.849145 w_k1_MSFT_s105
                                 + 0.8362 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.25905 w_k1_ADM_s105
                                 + 1.05699 w_k1_AXP_s105
                                 + 0.972758 w_k1_DIS_s105
                                 + 1.01791 w_k1_FDX_s105
                                 + 1.04868 w_k1_IBM_s105
                                 + 1.05617 w_k1_INTC_s105
                                 + 0.926662 w_k1_KO_s105
                                 + 0.807382 w_k1_MAT_s105
                                 + 1.05406 w_k1_MSFT_s105
                                 + 1.04917 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.10092 w_k1_ADM_s105
                                 + 0.877562 w_k1_AXP_s105
                                 + 1.10427 w_k1_DIS_s105
                                 + 1.05768 w_k1_FDX_s105
                                 + 1.57179 w_k1_IBM_s105
                                 + 0.959301 w_k1_INTC_s105
                                 + 1.0839 w_k1_KO_s105 + 1.11307 w_k1_MAT_s105
                                 + 0.920021 w_k1_MSFT_s105
                                 + 0.937072 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.807257 w_k1_ADM_s106
                                 + 1.07133 w_k1_AXP_s106
                                 + 0.798998 w_k1_DIS_s106
                                 + 0.821048 w_k1_FDX_s106
                                 + 1.01976 w_k1_IBM_s106
                                 + 0.93362 w_k1_INTC_s106
                                 + 0.816433 w_k1_KO_s106
                                 + 1.16765 w_k1_MAT_s106
                                 + 0.79366 w_k1_MSFT_s106
                                 + 0.902885 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.34828 w_k1_ADM_s106
                                 + 1.17199 w_k1_AXP_s106
                                 + 1.50695 w_k1_DIS_s106
                                 + 1.28271 w_k1_FDX_s106
                                 + 1.11777 w_k1_IBM_s106
                                 + 1.19469 w_k1_INTC_s106
                                 + 1.18959 w_k1_KO_s106 + 1.16186 w_k1_MAT_s106
                                 + 1.3694 w_k1_MSFT_s106 + 1.12832 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.07132 w_k1_ADM_s106
                                 + 1.31812 w_k1_AXP_s106
                                 + 1.04953 w_k1_DIS_s106
                                 + 1.07587 w_k1_FDX_s106
                                 + 1.19852 w_k1_IBM_s106
                                 + 1.00409 w_k1_INTC_s106
                                 + 0.970473 w_k1_KO_s106
                                 + 1.21935 w_k1_MAT_s106
                                 + 1.03043 w_k1_MSFT_s106
                                 + 1.07167 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.878428 w_k1_ADM_s106
                                 + 0.842929 w_k1_AXP_s106
                                 + 0.961687 w_k1_DIS_s106
                                 + 0.83233 w_k1_FDX_s106
                                 + 1.31958 w_k1_IBM_s106
                                 + 0.964294 w_k1_INTC_s106
                                 + 1.03167 w_k1_KO_s106 + 1.35689 w_k1_MAT_s106
                                 + 0.857749 w_k1_MSFT_s106
                                 + 0.932501 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.875101 w_k1_ADM_s106
                                 + 0.712369 w_k1_AXP_s106
                                 + 0.767296 w_k1_DIS_s106
                                 + 0.877077 w_k1_FDX_s106
                                 + 0.628957 w_k1_IBM_s106
                                 + 0.84334 w_k1_INTC_s106
                                 + 0.720994 w_k1_KO_s106
                                 + 0.618001 w_k1_MAT_s106
                                 + 0.820026 w_k1_MSFT_s106
                                 + 0.873201 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.20562 w_k1_ADM_s106
                                 + 1.11852 w_k1_AXP_s106
                                 + 0.988963 w_k1_DIS_s106
                                 + 1.10146 w_k1_FDX_s106
                                 + 0.80109 w_k1_IBM_s106
                                 + 1.16103 w_k1_INTC_s106
                                 + 0.833025 w_k1_KO_s106
                                 + 0.967927 w_k1_MAT_s106
                                 + 1.06817 w_k1_MSFT_s106
                                 + 0.960523 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.905234 w_k1_ADM_s106
                                 + 0.877813 w_k1_AXP_s106
                                 + 1.30495 w_k1_DIS_s106
                                 + 1.03859 w_k1_FDX_s106
                                 + 1.19286 w_k1_IBM_s106
                                 + 0.952886 w_k1_INTC_s106
                                 + 1.07521 w_k1_KO_s106 + 1.18649 w_k1_MAT_s106
                                 + 1.05408 w_k1_MSFT_s106
                                 + 1.04304 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.803461 w_k1_ADM_s106
                                 + 0.687649 w_k1_AXP_s106
                                 + 0.750196 w_k1_DIS_s106
                                 + 0.786589 w_k1_FDX_s106
                                 + 0.824958 w_k1_IBM_s106
                                 + 1.0143 w_k1_INTC_s106
                                 + 0.743764 w_k1_KO_s106
                                 + 0.708973 w_k1_MAT_s106
                                 + 0.829465 w_k1_MSFT_s106
                                 + 0.9058 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.952985 w_k1_ADM_s107
                                 + 1.05849 w_k1_AXP_s107
                                 + 0.848831 w_k1_DIS_s107
                                 + 0.965251 w_k1_FDX_s107
                                 + 0.811894 w_k1_IBM_s107
                                 + 0.938879 w_k1_INTC_s107
                                 + 0.748278 w_k1_KO_s107
                                 + 0.963342 w_k1_MAT_s107
                                 + 0.825959 w_k1_MSFT_s107
                                 + 0.916979 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.21422 w_k1_ADM_s107
                                 + 1.11712 w_k1_AXP_s107
                                 + 1.33817 w_k1_DIS_s107
                                 + 1.22195 w_k1_FDX_s107
                                 + 0.728338 w_k1_IBM_s107
                                 + 1.29957 w_k1_INTC_s107
                                 + 1.04099 w_k1_KO_s107
                                 + 0.660857 w_k1_MAT_s107
                                 + 1.13128 w_k1_MSFT_s107
                                 + 1.07888 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.827045 w_k1_ADM_s107
                                 + 0.979909 w_k1_AXP_s107
                                 + 1.21842 w_k1_DIS_s107
                                 + 0.996002 w_k1_FDX_s107
                                 + 0.958024 w_k1_IBM_s107
                                 + 1.1544 w_k1_INTC_s107
                                 + 0.949697 w_k1_KO_s107
                                 + 0.955468 w_k1_MAT_s107
                                 + 1.13228 w_k1_MSFT_s107
                                 + 1.09247 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.723996 w_k1_ADM_s107
                                 + 0.818195 w_k1_AXP_s107
                                 + 0.803284 w_k1_DIS_s107
                                 + 0.799208 w_k1_FDX_s107
                                 + 0.951201 w_k1_IBM_s107
                                 + 0.897721 w_k1_INTC_s107
                                 + 0.811688 w_k1_KO_s107
                                 + 0.802744 w_k1_MAT_s107
                                 + 0.796239 w_k1_MSFT_s107
                                 + 0.790124 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.81734 w_k1_ADM_s107
                                 + 0.69015 w_k1_AXP_s107
                                 + 0.66825 w_k1_DIS_s107
                                 + 0.784449 w_k1_FDX_s107
                                 + 0.591058 w_k1_IBM_s107
                                 + 0.860586 w_k1_INTC_s107
                                 + 0.625899 w_k1_KO_s107
                                 + 0.464408 w_k1_MAT_s107
                                 + 0.824251 w_k1_MSFT_s107
                                 + 0.819315 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.814686 w_k1_ADM_s107
                                 + 0.735901 w_k1_AXP_s107
                                 + 0.79283 w_k1_DIS_s107
                                 + 0.73827 w_k1_FDX_s107
                                 + 1.15026 w_k1_IBM_s107
                                 + 0.842208 w_k1_INTC_s107
                                 + 0.952787 w_k1_KO_s107
                                 + 0.810157 w_k1_MAT_s107
                                 + 0.787878 w_k1_MSFT_s107
                                 + 0.99103 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.995011 w_k1_ADM_s107
                                 + 1.30533 w_k1_AXP_s107 + 1.2704 w_k1_DIS_s107
                                 + 0.969631 w_k1_FDX_s107
                                 + 1.43984 w_k1_IBM_s107
                                 + 1.19792 w_k1_INTC_s107
                                 + 1.26066 w_k1_KO_s107 + 1.41405 w_k1_MAT_s107
                                 + 0.953742 w_k1_MSFT_s107
                                 + 1.14651 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.28554 w_k1_ADM_s107
                                 + 1.42746 w_k1_AXP_s107
                                 + 1.85738 w_k1_DIS_s107
                                 + 1.36342 w_k1_FDX_s107 + 1.2527 w_k1_IBM_s107
                                 + 1.36898 w_k1_INTC_s107
                                 + 1.25495 w_k1_KO_s107 + 1.38771 w_k1_MAT_s107
                                 + 1.23878 w_k1_MSFT_s107
                                 + 1.21028 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.93977 w_k1_ADM_s110
                                 + 1.10787 w_k1_AXP_s110
                                 + 1.05806 w_k1_DIS_s110
                                 + 1.20606 w_k1_FDX_s110
                                 + 0.866182 w_k1_IBM_s110
                                 + 1.19021 w_k1_INTC_s110
                                 + 1.04394 w_k1_KO_s110
                                 + 0.653998 w_k1_MAT_s110
                                 + 1.04798 w_k1_MSFT_s110
                                 + 0.951682 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.25523 w_k1_ADM_s110
                                 + 1.29275 w_k1_AXP_s110
                                 + 1.41028 w_k1_DIS_s110
                                 + 1.41577 w_k1_FDX_s110
                                 + 0.90286 w_k1_IBM_s110
                                 + 1.18466 w_k1_INTC_s110
                                 + 1.17112 w_k1_KO_s110
                                 + 0.811737 w_k1_MAT_s110
                                 + 1.12799 w_k1_MSFT_s110
                                 + 1.12268 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.86782 w_k1_ADM_s110
                                 + 1.17546 w_k1_AXP_s110
                                 + 0.837328 w_k1_DIS_s110
                                 + 0.987716 w_k1_FDX_s110
                                 + 1.19556 w_k1_IBM_s110
                                 + 0.809801 w_k1_INTC_s110
                                 + 0.938348 w_k1_KO_s110
                                 + 1.27359 w_k1_MAT_s110
                                 + 0.852755 w_k1_MSFT_s110
                                 + 0.885072 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.17425 w_k1_ADM_s110
                                 + 1.64608 w_k1_AXP_s110
                                 + 1.73707 w_k1_DIS_s110
                                 + 1.39603 w_k1_FDX_s110
                                 + 1.53849 w_k1_IBM_s110
                                 + 1.1209 w_k1_INTC_s110 + 1.41197 w_k1_KO_s110
                                 + 1.0093 w_k1_MAT_s110
                                 + 1.15124 w_k1_MSFT_s110 + 1.1535 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.892836 w_k1_ADM_s110
                                 + 0.971572 w_k1_AXP_s110
                                 + 1.0848 w_k1_DIS_s110 + 1.03128 w_k1_FDX_s110
                                 + 0.981533 w_k1_IBM_s110
                                 + 1.10441 w_k1_INTC_s110
                                 + 1.11747 w_k1_KO_s110
                                 + 0.794337 w_k1_MAT_s110
                                 + 1.03368 w_k1_MSFT_s110
                                 + 0.973173 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.0156 w_k1_ADM_s110
                                 + 0.89968 w_k1_AXP_s110
                                 + 1.12605 w_k1_DIS_s110
                                 + 0.901955 w_k1_FDX_s110 + 1.561 w_k1_IBM_s110
                                 + 0.859106 w_k1_INTC_s110
                                 + 1.22009 w_k1_KO_s110
                                 + 0.918705 w_k1_MAT_s110
                                 + 0.967566 w_k1_MSFT_s110
                                 + 0.938428 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.02116 w_k1_ADM_s110
                                 + 1.25758 w_k1_AXP_s110
                                 + 1.33065 w_k1_DIS_s110
                                 + 1.21286 w_k1_FDX_s110
                                 + 1.34455 w_k1_IBM_s110
                                 + 1.22977 w_k1_INTC_s110
                                 + 1.20501 w_k1_KO_s110 + 1.00131 w_k1_MAT_s110
                                 + 0.98763 w_k1_MSFT_s110
                                 + 1.10999 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.762249 w_k1_ADM_s110
                                 + 0.780787 w_k1_AXP_s110
                                 + 0.733986 w_k1_DIS_s110
                                 + 0.824436 w_k1_FDX_s110
                                 + 0.841126 w_k1_IBM_s110
                                 + 0.861501 w_k1_INTC_s110
                                 + 0.728719 w_k1_KO_s110
                                 + 0.632101 w_k1_MAT_s110
                                 + 0.748584 w_k1_MSFT_s110
                                 + 0.772201 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.14659 w_k1_ADM_s111
                                 + 0.92244 w_k1_AXP_s111
                                 + 0.965334 w_k1_DIS_s111
                                 + 1.08392 w_k1_FDX_s111
                                 + 0.750274 w_k1_IBM_s111
                                 + 1.42881 w_k1_INTC_s111
                                 + 1.03951 w_k1_KO_s111
                                 + 0.385871 w_k1_MAT_s111
                                 + 1.06563 w_k1_MSFT_s111
                                 + 1.05534 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.0306 w_k1_ADM_s111
                                 + 1.15532 w_k1_AXP_s111 + 1.0952 w_k1_DIS_s111
                                 + 1.1545 w_k1_FDX_s111
                                 + 0.923227 w_k1_IBM_s111
                                 + 1.33622 w_k1_INTC_s111
                                 + 1.07948 w_k1_KO_s111
                                 + 0.571001 w_k1_MAT_s111
                                 + 1.14655 w_k1_MSFT_s111
                                 + 1.12216 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.9621 w_k1_ADM_s111
                                 + 1.1251 w_k1_AXP_s111 + 1.26083 w_k1_DIS_s111
                                 + 1.07337 w_k1_FDX_s111
                                 + 1.57772 w_k1_IBM_s111
                                 + 1.33435 w_k1_INTC_s111
                                 + 1.47336 w_k1_KO_s111 + 1.14651 w_k1_MAT_s111
                                 + 1.10281 w_k1_MSFT_s111
                                 + 1.11701 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.12726 w_k1_ADM_s111
                                 + 1.44835 w_k1_AXP_s111
                                 + 1.29246 w_k1_DIS_s111
                                 + 1.24591 w_k1_FDX_s111
                                 + 1.08036 w_k1_IBM_s111
                                 + 1.31979 w_k1_INTC_s111
                                 + 1.47518 w_k1_KO_s111 + 1.22774 w_k1_MAT_s111
                                 + 1.23248 w_k1_MSFT_s111
                                 + 1.17466 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.53736 w_k1_ADM_s111
                                 + 1.9296 w_k1_AXP_s111 + 1.82392 w_k1_DIS_s111
                                 + 1.42366 w_k1_FDX_s111
                                 + 1.69241 w_k1_IBM_s111
                                 + 1.34187 w_k1_INTC_s111
                                 + 1.69461 w_k1_KO_s111 + 1.53056 w_k1_MAT_s111
                                 + 1.23138 w_k1_MSFT_s111 + 1.3228 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.916689 w_k1_ADM_s111
                                 + 0.769757 w_k1_AXP_s111
                                 + 0.707632 w_k1_DIS_s111
                                 + 0.814922 w_k1_FDX_s111
                                 + 0.708238 w_k1_IBM_s111
                                 + 0.878472 w_k1_INTC_s111
                                 + 0.915707 w_k1_KO_s111
                                 + 0.627353 w_k1_MAT_s111
                                 + 0.810357 w_k1_MSFT_s111
                                 + 0.859399 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.36592 w_k1_ADM_s111
                                 + 1.48326 w_k1_AXP_s111
                                 + 1.85275 w_k1_DIS_s111
                                 + 1.71426 w_k1_FDX_s111
                                 + 1.10668 w_k1_IBM_s111
                                 + 1.28849 w_k1_INTC_s111
                                 + 1.56914 w_k1_KO_s111
                                 + 0.860784 w_k1_MAT_s111
                                 + 1.31718 w_k1_MSFT_s111
                                 + 1.18747 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.719357 w_k1_ADM_s111
                                 + 0.81261 w_k1_AXP_s111
                                 + 0.814314 w_k1_DIS_s111
                                 + 0.875223 w_k1_FDX_s111
                                 + 1.0451 w_k1_IBM_s111
                                 + 1.03724 w_k1_INTC_s111
                                 + 0.993201 w_k1_KO_s111
                                 + 0.693503 w_k1_MAT_s111
                                 + 0.903354 w_k1_MSFT_s111
                                 + 0.941381 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.78031 w_k1_ADM_s112
                                 + 1.08925 w_k1_AXP_s112
                                 + 0.805578 w_k1_DIS_s112
                                 + 0.891078 w_k1_FDX_s112
                                 + 1.26478 w_k1_IBM_s112
                                 + 1.02254 w_k1_INTC_s112
                                 + 0.896589 w_k1_KO_s112
                                 + 1.31431 w_k1_MAT_s112
                                 + 0.89213 w_k1_MSFT_s112
                                 + 0.964234 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.844676 w_k1_ADM_s112
                                 + 0.763176 w_k1_AXP_s112
                                 + 0.623065 w_k1_DIS_s112
                                 + 0.925743 w_k1_FDX_s112
                                 + 0.634834 w_k1_IBM_s112
                                 + 0.910824 w_k1_INTC_s112
                                 + 0.591668 w_k1_KO_s112
                                 + 0.504373 w_k1_MAT_s112
                                 + 0.913134 w_k1_MSFT_s112
                                 + 0.835092 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.09857 w_k1_ADM_s112
                                 + 0.951141 w_k1_AXP_s112
                                 + 1.1379 w_k1_DIS_s112 + 1.23043 w_k1_FDX_s112
                                 + 1.04051 w_k1_IBM_s112
                                 + 0.991804 w_k1_INTC_s112
                                 + 1.00522 w_k1_KO_s112 + 1.18838 w_k1_MAT_s112
                                 + 1.16136 w_k1_MSFT_s112
                                 + 0.982966 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.848888 w_k1_ADM_s112
                                 + 1.47022 w_k1_AXP_s112
                                 + 1.42837 w_k1_DIS_s112
                                 + 1.32231 w_k1_FDX_s112
                                 + 1.81343 w_k1_IBM_s112
                                 + 1.31343 w_k1_INTC_s112
                                 + 1.34427 w_k1_KO_s112 + 2.07109 w_k1_MAT_s112
                                 + 1.31144 w_k1_MSFT_s112
                                 + 1.22688 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.02593 w_k1_ADM_s112
                                 + 1.20895 w_k1_AXP_s112
                                 + 0.984473 w_k1_DIS_s112
                                 + 0.991025 w_k1_FDX_s112
                                 + 1.65728 w_k1_IBM_s112
                                 + 0.865095 w_k1_INTC_s112
                                 + 1.03387 w_k1_KO_s112 + 1.90296 w_k1_MAT_s112
                                 + 0.889101 w_k1_MSFT_s112
                                 + 0.998622 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.787198 w_k1_ADM_s112
                                 + 0.512969 w_k1_AXP_s112
                                 + 0.551574 w_k1_DIS_s112
                                 + 0.707058 w_k1_FDX_s112
                                 + 0.562629 w_k1_IBM_s112
                                 + 0.755383 w_k1_INTC_s112
                                 + 0.583944 w_k1_KO_s112
                                 + 0.503027 w_k1_MAT_s112
                                 + 0.755772 w_k1_MSFT_s112
                                 + 0.754146 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.956296 w_k1_ADM_s112
                                 + 0.832992 w_k1_AXP_s112
                                 + 0.825502 w_k1_DIS_s112
                                 + 0.964809 w_k1_FDX_s112
                                 + 0.907909 w_k1_IBM_s112
                                 + 0.90155 w_k1_INTC_s112
                                 + 0.771374 w_k1_KO_s112
                                 + 0.83785 w_k1_MAT_s112
                                 + 0.861929 w_k1_MSFT_s112
                                 + 0.858412 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.05997 w_k1_ADM_s112
                                 + 1.63783 w_k1_AXP_s112
                                 + 1.15926 w_k1_DIS_s112
                                 + 1.27864 w_k1_FDX_s112
                                 + 1.24288 w_k1_IBM_s112
                                 + 1.12358 w_k1_INTC_s112
                                 + 1.01015 w_k1_KO_s112 + 1.60148 w_k1_MAT_s112
                                 + 1.13407 w_k1_MSFT_s112
                                 + 1.10611 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.26981 w_k1_ADM_s113
                                 + 1.73316 w_k1_AXP_s113
                                 + 1.62941 w_k1_DIS_s113
                                 + 1.35859 w_k1_FDX_s113
                                 + 1.73585 w_k1_IBM_s113
                                 + 1.24448 w_k1_INTC_s113
                                 + 1.61929 w_k1_KO_s113 + 2.85403 w_k1_MAT_s113
                                 + 1.20786 w_k1_MSFT_s113
                                 + 1.21393 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.698 w_k1_ADM_s113
                                 + 0.705509 w_k1_AXP_s113
                                 + 0.619432 w_k1_DIS_s113
                                 + 0.818368 w_k1_FDX_s113
                                 + 0.662989 w_k1_IBM_s113
                                 + 0.923023 w_k1_INTC_s113
                                 + 0.819389 w_k1_KO_s113
                                 + 0.951591 w_k1_MAT_s113
                                 + 0.86875 w_k1_MSFT_s113
                                 + 0.76115 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.861526 w_k1_ADM_s113
                                 + 1.1418 w_k1_AXP_s113 + 1.10146 w_k1_DIS_s113
                                 + 1.12624 w_k1_FDX_s113
                                 + 0.832048 w_k1_IBM_s113
                                 + 1.13585 w_k1_INTC_s113
                                 + 1.04071 w_k1_KO_s113 + 1.54594 w_k1_MAT_s113
                                 + 1.11988 w_k1_MSFT_s113
                                 + 1.00443 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.932994 w_k1_ADM_s113
                                 + 1.34306 w_k1_AXP_s113
                                 + 1.16423 w_k1_DIS_s113
                                 + 1.12693 w_k1_FDX_s113
                                 + 1.25915 w_k1_IBM_s113
                                 + 1.03804 w_k1_INTC_s113
                                 + 1.15057 w_k1_KO_s113 + 1.90732 w_k1_MAT_s113
                                 + 1.1178 w_k1_MSFT_s113 + 1.05156 w_k1_PG_s113
                                 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.929433 w_k1_ADM_s113
                                 + 0.732124 w_k1_AXP_s113
                                 + 0.813988 w_k1_DIS_s113
                                 + 0.857572 w_k1_FDX_s113
                                 + 0.881144 w_k1_IBM_s113
                                 + 0.93796 w_k1_INTC_s113
                                 + 0.969799 w_k1_KO_s113
                                 + 1.09973 w_k1_MAT_s113
                                 + 0.920987 w_k1_MSFT_s113
                                 + 0.882909 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.961329 w_k1_ADM_s113
                                 + 0.856696 w_k1_AXP_s113
                                 + 0.998869 w_k1_DIS_s113
                                 + 0.969296 w_k1_FDX_s113
                                 + 1.13651 w_k1_IBM_s113
                                 + 1.05152 w_k1_INTC_s113
                                 + 1.18153 w_k1_KO_s113
                                 + 0.990405 w_k1_MAT_s113
                                 + 1.04132 w_k1_MSFT_s113
                                 + 1.01373 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.84218 w_k1_ADM_s113
                                 + 0.783757 w_k1_AXP_s113
                                 + 0.758472 w_k1_DIS_s113
                                 + 0.775701 w_k1_FDX_s113 + 1.183 w_k1_IBM_s113
                                 + 1.08589 w_k1_INTC_s113
                                 + 0.970879 w_k1_KO_s113
                                 + 0.971256 w_k1_MAT_s113
                                 + 0.868082 w_k1_MSFT_s113
                                 + 0.967646 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.30869 w_k1_ADM_s113
                                 + 1.35059 w_k1_AXP_s113
                                 + 1.67058 w_k1_DIS_s113
                                 + 1.44475 w_k1_FDX_s113
                                 + 1.00937 w_k1_IBM_s113
                                 + 1.20248 w_k1_INTC_s113
                                 + 1.17996 w_k1_KO_s113 + 1.11715 w_k1_MAT_s113
                                 + 1.27457 w_k1_MSFT_s113
                                 + 1.19683 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.36629 w_k1_ADM_s114
                                 + 1.85787 w_k1_AXP_s114
                                 + 1.94948 w_k1_DIS_s114 + 1.6149 w_k1_FDX_s114
                                 + 1.78177 w_k1_IBM_s114
                                 + 1.12195 w_k1_INTC_s114
                                 + 1.64876 w_k1_KO_s114 + 2.05477 w_k1_MAT_s114
                                 + 1.39401 w_k1_MSFT_s114
                                 + 1.49527 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.981193 w_k1_ADM_s114
                                 + 1.21534 w_k1_AXP_s114
                                 + 0.966131 w_k1_DIS_s114
                                 + 1.08137 w_k1_FDX_s114
                                 + 0.881195 w_k1_IBM_s114
                                 + 0.894455 w_k1_INTC_s114
                                 + 0.887414 w_k1_KO_s114
                                 + 0.931085 w_k1_MAT_s114
                                 + 1.01781 w_k1_MSFT_s114
                                 + 1.02398 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.779395 w_k1_ADM_s114
                                 + 0.753085 w_k1_AXP_s114
                                 + 0.642921 w_k1_DIS_s114
                                 + 0.821145 w_k1_FDX_s114
                                 + 0.802887 w_k1_IBM_s114
                                 + 0.863826 w_k1_INTC_s114
                                 + 0.845542 w_k1_KO_s114
                                 + 0.70854 w_k1_MAT_s114
                                 + 0.778236 w_k1_MSFT_s114
                                 + 0.949714 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.658444 w_k1_ADM_s114
                                 + 1.05469 w_k1_AXP_s114
                                 + 0.93676 w_k1_DIS_s114
                                 + 0.995981 w_k1_FDX_s114
                                 + 1.00495 w_k1_IBM_s114
                                 + 1.01944 w_k1_INTC_s114
                                 + 1.02147 w_k1_KO_s114 + 1.29119 w_k1_MAT_s114
                                 + 0.974272 w_k1_MSFT_s114
                                 + 1.07024 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.96037 w_k1_ADM_s114
                                 + 1.03241 w_k1_AXP_s114
                                 + 1.19664 w_k1_DIS_s114
                                 + 1.07413 w_k1_FDX_s114
                                 + 1.17282 w_k1_IBM_s114
                                 + 1.00217 w_k1_INTC_s114
                                 + 1.16143 w_k1_KO_s114 + 1.09429 w_k1_MAT_s114
                                 + 1.16718 w_k1_MSFT_s114
                                 + 1.08726 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.861283 w_k1_ADM_s114
                                 + 0.833297 w_k1_AXP_s114
                                 + 1.15763 w_k1_DIS_s114
                                 + 1.05545 w_k1_FDX_s114
                                 + 0.85968 w_k1_IBM_s114
                                 + 0.948396 w_k1_INTC_s114
                                 + 1.15072 w_k1_KO_s114
                                 + 0.809511 w_k1_MAT_s114
                                 + 1.00043 w_k1_MSFT_s114
                                 + 1.03709 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.918476 w_k1_ADM_s114
                                 + 0.796782 w_k1_AXP_s114
                                 + 0.798029 w_k1_DIS_s114
                                 + 0.827883 w_k1_FDX_s114
                                 + 1.23613 w_k1_IBM_s114
                                 + 0.808584 w_k1_INTC_s114
                                 + 1.00561 w_k1_KO_s114 + 1.29558 w_k1_MAT_s114
                                 + 0.80043 w_k1_MSFT_s114
                                 + 1.03067 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.36254 w_k1_ADM_s114
                                 + 1.58838 w_k1_AXP_s114
                                 + 1.52142 w_k1_DIS_s114
                                 + 1.36075 w_k1_FDX_s114
                                 + 1.31648 w_k1_IBM_s114
                                 + 1.21152 w_k1_INTC_s114
                                 + 1.21281 w_k1_KO_s114 + 1.51052 w_k1_MAT_s114
                                 + 1.29297 w_k1_MSFT_s114
                                 + 1.16515 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.12622 w_k1_ADM_s115
                                 + 1.02103 w_k1_AXP_s115
                                 + 1.06736 w_k1_DIS_s115
                                 + 1.07405 w_k1_FDX_s115
                                 + 1.46929 w_k1_IBM_s115
                                 + 0.897959 w_k1_INTC_s115
                                 + 1.15555 w_k1_KO_s115 + 1.21886 w_k1_MAT_s115
                                 + 0.907906 w_k1_MSFT_s115
                                 + 0.976346 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.13358 w_k1_ADM_s115
                                 + 1.07178 w_k1_AXP_s115
                                 + 1.62654 w_k1_DIS_s115 + 1.3131 w_k1_FDX_s115
                                 + 1.14549 w_k1_IBM_s115
                                 + 1.20071 w_k1_INTC_s115
                                 + 1.18105 w_k1_KO_s115 + 1.05752 w_k1_MAT_s115
                                 + 1.3483 w_k1_MSFT_s115 + 1.25842 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.878499 w_k1_ADM_s115
                                 + 0.917177 w_k1_AXP_s115
                                 + 0.863277 w_k1_DIS_s115
                                 + 0.91906 w_k1_FDX_s115
                                 + 1.13991 w_k1_IBM_s115
                                 + 0.935737 w_k1_INTC_s115
                                 + 0.863826 w_k1_KO_s115
                                 + 0.731225 w_k1_MAT_s115
                                 + 0.851072 w_k1_MSFT_s115
                                 + 0.919733 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.1199 w_k1_ADM_s115
                                 + 0.785202 w_k1_AXP_s115
                                 + 1.0589 w_k1_DIS_s115 + 1.16671 w_k1_FDX_s115
                                 + 0.738097 w_k1_IBM_s115
                                 + 0.988251 w_k1_INTC_s115
                                 + 0.828334 w_k1_KO_s115
                                 + 0.748264 w_k1_MAT_s115
                                 + 1.16711 w_k1_MSFT_s115
                                 + 0.893437 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.08392 w_k1_ADM_s115
                                 + 1.10392 w_k1_AXP_s115
                                 + 1.06385 w_k1_DIS_s115
                                 + 1.07852 w_k1_FDX_s115
                                 + 0.882572 w_k1_IBM_s115
                                 + 1.11787 w_k1_INTC_s115
                                 + 1.02622 w_k1_KO_s115
                                 + 0.887142 w_k1_MAT_s115
                                 + 1.12068 w_k1_MSFT_s115
                                 + 0.889146 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.10967 w_k1_ADM_s115
                                 + 0.65018 w_k1_AXP_s115
                                 + 1.09526 w_k1_DIS_s115
                                 + 0.945064 w_k1_FDX_s115
                                 + 0.790574 w_k1_IBM_s115
                                 + 0.807051 w_k1_INTC_s115
                                 + 0.860185 w_k1_KO_s115
                                 + 0.528853 w_k1_MAT_s115
                                 + 1.05236 w_k1_MSFT_s115 + 1.0024 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.648266 w_k1_ADM_s115
                                 + 0.580626 w_k1_AXP_s115
                                 + 0.670796 w_k1_DIS_s115
                                 + 0.733298 w_k1_FDX_s115
                                 + 0.886486 w_k1_IBM_s115
                                 + 0.615529 w_k1_INTC_s115
                                 + 0.73629 w_k1_KO_s115
                                 + 0.766237 w_k1_MAT_s115
                                 + 0.742667 w_k1_MSFT_s115
                                 + 0.897822 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.06976 w_k1_ADM_s115
                                 + 1.29356 w_k1_AXP_s115
                                 + 1.29184 w_k1_DIS_s115
                                 + 1.11326 w_k1_FDX_s115
                                 + 1.49501 w_k1_IBM_s115
                                 + 1.10022 w_k1_INTC_s115
                                 + 1.39492 w_k1_KO_s115 + 1.90641 w_k1_MAT_s115
                                 + 1.21588 w_k1_MSFT_s115
                                 + 1.16632 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.908619 w_k1_ADM_s116
                                 + 0.954985 w_k1_AXP_s116
                                 + 0.975009 w_k1_DIS_s116
                                 + 0.970079 w_k1_FDX_s116
                                 + 1.41823 w_k1_IBM_s116
                                 + 0.914827 w_k1_INTC_s116
                                 + 1.11949 w_k1_KO_s116 + 1.46577 w_k1_MAT_s116
                                 + 0.941645 w_k1_MSFT_s116
                                 + 0.995792 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.42924 w_k1_ADM_s116
                                 + 1.20272 w_k1_AXP_s116
                                 + 1.48085 w_k1_DIS_s116
                                 + 1.24095 w_k1_FDX_s116
                                 + 1.20111 w_k1_IBM_s116
                                 + 1.3387 w_k1_INTC_s116 + 1.42086 w_k1_KO_s116
                                 + 0.83939 w_k1_MAT_s116
                                 + 1.15489 w_k1_MSFT_s116
                                 + 1.17392 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.825283 w_k1_ADM_s116
                                 + 0.684634 w_k1_AXP_s116
                                 + 0.541913 w_k1_DIS_s116
                                 + 0.692304 w_k1_FDX_s116
                                 + 0.683509 w_k1_IBM_s116
                                 + 0.823433 w_k1_INTC_s116
                                 + 0.713376 w_k1_KO_s116
                                 + 0.75452 w_k1_MAT_s116
                                 + 0.67875 w_k1_MSFT_s116
                                 + 0.756933 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.24925 w_k1_ADM_s116
                                 + 1.0166 w_k1_AXP_s116 + 1.23422 w_k1_DIS_s116
                                 + 1.23829 w_k1_FDX_s116
                                 + 0.829435 w_k1_IBM_s116
                                 + 1.1219 w_k1_INTC_s116 + 1.18295 w_k1_KO_s116
                                 + 0.750508 w_k1_MAT_s116
                                 + 1.36749 w_k1_MSFT_s116
                                 + 1.08969 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.774737 w_k1_ADM_s116
                                 + 0.745045 w_k1_AXP_s116
                                 + 0.725503 w_k1_DIS_s116
                                 + 0.874962 w_k1_FDX_s116
                                 + 0.750267 w_k1_IBM_s116
                                 + 1.08803 w_k1_INTC_s116
                                 + 0.931457 w_k1_KO_s116
                                 + 0.753969 w_k1_MAT_s116
                                 + 0.943194 w_k1_MSFT_s116
                                 + 0.853614 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.12011 w_k1_ADM_s116
                                 + 1.0447 w_k1_AXP_s116
                                 + 0.982903 w_k1_DIS_s116
                                 + 1.07114 w_k1_FDX_s116
                                 + 0.934025 w_k1_IBM_s116
                                 + 0.901984 w_k1_INTC_s116
                                 + 1.02939 w_k1_KO_s116 + 0.84479 w_k1_MAT_s116
                                 + 1.03917 w_k1_MSFT_s116
                                 + 0.909252 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.966039 w_k1_ADM_s116
                                 + 0.782482 w_k1_AXP_s116
                                 + 0.902743 w_k1_DIS_s116
                                 + 0.894563 w_k1_FDX_s116
                                 + 0.944801 w_k1_IBM_s116
                                 + 0.922256 w_k1_INTC_s116
                                 + 1.22368 w_k1_KO_s116
                                 + 0.864426 w_k1_MAT_s116
                                 + 0.914285 w_k1_MSFT_s116
                                 + 1.0044 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.10042 w_k1_ADM_s116
                                 + 0.95016 w_k1_AXP_s116
                                 + 0.868536 w_k1_DIS_s116
                                 + 1.09002 w_k1_FDX_s116
                                 + 0.695666 w_k1_IBM_s116
                                 + 0.910029 w_k1_INTC_s116
                                 + 0.822923 w_k1_KO_s116
                                 + 0.669408 w_k1_MAT_s116
                                 + 0.886451 w_k1_MSFT_s116
                                 + 0.884644 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.3006 w_k1_ADM_s117
                                 + 0.975142 w_k1_AXP_s117
                                 + 1.11934 w_k1_DIS_s117
                                 + 1.04451 w_k1_FDX_s117
                                 + 1.59583 w_k1_IBM_s117
                                 + 1.01083 w_k1_INTC_s117
                                 + 1.34003 w_k1_KO_s117 + 1.51949 w_k1_MAT_s117
                                 + 1.14736 w_k1_MSFT_s117
                                 + 1.22804 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.02571 w_k1_ADM_s117
                                 + 0.634593 w_k1_AXP_s117
                                 + 0.76799 w_k1_DIS_s117
                                 + 0.793907 w_k1_FDX_s117
                                 + 0.863648 w_k1_IBM_s117
                                 + 0.798801 w_k1_INTC_s117
                                 + 0.828154 w_k1_KO_s117
                                 + 0.719145 w_k1_MAT_s117
                                 + 0.899592 w_k1_MSFT_s117
                                 + 0.84292 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.814157 w_k1_ADM_s117
                                 + 0.696682 w_k1_AXP_s117
                                 + 0.485056 w_k1_DIS_s117
                                 + 0.667097 w_k1_FDX_s117
                                 + 0.718088 w_k1_IBM_s117
                                 + 0.708038 w_k1_INTC_s117
                                 + 0.636096 w_k1_KO_s117
                                 + 0.975429 w_k1_MAT_s117
                                 + 0.656178 w_k1_MSFT_s117
                                 + 0.72463 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.834102 w_k1_ADM_s117
                                 + 0.798097 w_k1_AXP_s117
                                 + 0.722024 w_k1_DIS_s117
                                 + 0.926538 w_k1_FDX_s117
                                 + 0.741172 w_k1_IBM_s117
                                 + 0.995908 w_k1_INTC_s117
                                 + 0.713572 w_k1_KO_s117
                                 + 0.812993 w_k1_MAT_s117
                                 + 1.01908 w_k1_MSFT_s117
                                 + 0.805305 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.892413 w_k1_ADM_s117
                                 + 0.935425 w_k1_AXP_s117
                                 + 0.801683 w_k1_DIS_s117
                                 + 0.819511 w_k1_FDX_s117
                                 + 1.21828 w_k1_IBM_s117
                                 + 0.837337 w_k1_INTC_s117
                                 + 0.919012 w_k1_KO_s117
                                 + 1.39272 w_k1_MAT_s117
                                 + 0.892362 w_k1_MSFT_s117
                                 + 0.90359 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.09775 w_k1_ADM_s117
                                 + 0.946748 w_k1_AXP_s117
                                 + 1.12021 w_k1_DIS_s117
                                 + 1.11317 w_k1_FDX_s117
                                 + 0.92521 w_k1_IBM_s117
                                 + 0.876155 w_k1_INTC_s117
                                 + 0.97961 w_k1_KO_s117 + 1.23553 w_k1_MAT_s117
                                 + 1.07297 w_k1_MSFT_s117
                                 + 1.00064 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.29469 w_k1_ADM_s117
                                 + 1.46039 w_k1_AXP_s117
                                 + 0.985985 w_k1_DIS_s117
                                 + 1.13504 w_k1_FDX_s117
                                 + 1.30797 w_k1_IBM_s117
                                 + 0.888762 w_k1_INTC_s117
                                 + 0.964968 w_k1_KO_s117
                                 + 1.98963 w_k1_MAT_s117
                                 + 1.06557 w_k1_MSFT_s117
                                 + 1.07444 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.76212 w_k1_ADM_s117
                                 + 1.13362 w_k1_AXP_s117 + 1.6613 w_k1_DIS_s117
                                 + 1.43283 w_k1_FDX_s117
                                 + 1.07775 w_k1_IBM_s117
                                 + 1.10263 w_k1_INTC_s117
                                 + 1.22286 w_k1_KO_s117 + 1.31609 w_k1_MAT_s117
                                 + 1.41572 w_k1_MSFT_s117
                                 + 1.11199 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0345909307262706 w_k1_ADM_s000
                                 + 0.0154103231532878 w_k1_ADM_s001
                                 + 0.00924618763692246 w_k1_ADM_s002
                                 + 0.0132155093125847 w_k1_ADM_s003
                                 + 0.0146378969172066 w_k1_ADM_s004
                                 + 0.0201694986157239 w_k1_ADM_s005
                                 + 0.018152023411716 w_k1_ADM_s006
                                 + 0.0435545567660976 w_k1_ADM_s007
                                 + 0.0176444900224611 w_k1_ADM_s010
                                 + 0.0296379257435722 w_k1_ADM_s011
                                 + 0.0299655870158435 w_k1_ADM_s012
                                 + 0.0105109117673552 w_k1_ADM_s013
                                 + 0.0223028205545544 w_k1_ADM_s014
                                 + 0.0208088245230857 w_k1_ADM_s015
                                 + 0.0396465842413942 w_k1_ADM_s016
                                 + 0.0202514984270149 w_k1_ADM_s017
                                 + 0.0324764517763644 w_k1_ADM_s100
                                 + 0.0559668312141216 w_k1_ADM_s101
                                 + 0.0502855027091623 w_k1_ADM_s102
                                 + 0.124468507517217 w_k1_ADM_s103
                                 + 0.0337143421785957 w_k1_ADM_s104
                                 + 0.0294147624001795 w_k1_ADM_s105
                                 + 0.0371795661516461 w_k1_ADM_s106
                                 + 0.0671763881920397 w_k1_ADM_s107
                                 + 0.035427989785506 w_k1_ADM_s110
                                 + 0.0190459332710902 w_k1_ADM_s111
                                 + 0.0533240784500287 w_k1_ADM_s112
                                 + 0.019827011801533 w_k1_ADM_s113
                                 + 0.0118892797135021 w_k1_ADM_s114
                                 + 0.0160352416124386 w_k1_ADM_s115
                                 + 0.045481718053709 w_k1_ADM_s116
                                 + 0.019218918293781 w_k1_ADM_s117
                                 + 0.0393266323199049 w_k1_AXP_s000
                                 + 0.0145947459143404 w_k1_AXP_s001
                                 + 0.00993951548717815 w_k1_AXP_s002
                                 + 0.0137557015259994 w_k1_AXP_s003
                                 + 0.014124438800203 w_k1_AXP_s004
                                 + 0.0209173576060969 w_k1_AXP_s005
                                 + 0.0188571481276285 w_k1_AXP_s006
                                 + 0.0395117296376886 w_k1_AXP_s007
                                 + 0.0179597797875698 w_k1_AXP_s010
                                 + 0.0293532292075252 w_k1_AXP_s011
                                 + 0.0293007327284181 w_k1_AXP_s012
                                 + 0.0104392918358414 w_k1_AXP_s013
                                 + 0.0231096299814906 w_k1_AXP_s014
                                 + 0.0200750231925801 w_k1_AXP_s015
                                 + 0.0439311411108896 w_k1_AXP_s016
                                 + 0.02177316074421 w_k1_AXP_s017
                                 + 0.0303480675769878 w_k1_AXP_s100
                                 + 0.0580608038171961 w_k1_AXP_s101
                                 + 0.0580330751933127 w_k1_AXP_s102
                                 + 0.117235901387678 w_k1_AXP_s103
                                 + 0.037111987324295 w_k1_AXP_s104
                                 + 0.0220406962776249 w_k1_AXP_s105
                                 + 0.0358655062239642 w_k1_AXP_s106
                                 + 0.0738399907023493 w_k1_AXP_s107
                                 + 0.0415241778182704 w_k1_AXP_s110
                                 + 0.0202511389520874 w_k1_AXP_s111
                                 + 0.0568530002901707 w_k1_AXP_s112
                                 + 0.0212595496632379 w_k1_AXP_s113
                                 + 0.012899472433162 w_k1_AXP_s114
                                 + 0.0146300421867279 w_k1_AXP_s115
                                 + 0.041051299808526 w_k1_AXP_s116
                                 + 0.0158888173411596 w_k1_AXP_s117
                                 + 0.039336219162787 w_k1_DIS_s000
                                 + 0.0157832960024421 w_k1_DIS_s001
                                 + 0.00994807310388706 w_k1_DIS_s002
                                 + 0.0156879882848873 w_k1_DIS_s003
                                 + 0.0149400736458283 w_k1_DIS_s004
                                 + 0.018556738929305 w_k1_DIS_s005
                                 + 0.0165032867514402 w_k1_DIS_s006
                                 + 0.0415731444835797 w_k1_DIS_s007
                                 + 0.0182268155525802 w_k1_DIS_s010
                                 + 0.0304420758709558 w_k1_DIS_s011
                                 + 0.0311985525699692 w_k1_DIS_s012
                                 + 0.0101092034627742 w_k1_DIS_s013
                                 + 0.0222575412537292 w_k1_DIS_s014
                                 + 0.0216161917180594 w_k1_DIS_s015
                                 + 0.043464752650876 w_k1_DIS_s016
                                 + 0.0219893601955905 w_k1_DIS_s017
                                 + 0.0311939735077424 w_k1_DIS_s100
                                 + 0.0554280870657591 w_k1_DIS_s101
                                 + 0.0598202346567615 w_k1_DIS_s102
                                 + 0.10986747551585 w_k1_DIS_s103
                                 + 0.034169581191145 w_k1_DIS_s104
                                 + 0.0238151856574359 w_k1_DIS_s105
                                 + 0.0390905096116329 w_k1_DIS_s106
                                 + 0.0801076605009407 w_k1_DIS_s107
                                 + 0.0395050172463609 w_k1_DIS_s110
                                 + 0.0203842892032671 w_k1_DIS_s111
                                 + 0.0524526678549085 w_k1_DIS_s112
                                 + 0.0218539640140404 w_k1_DIS_s113
                                 + 0.0129650252724296 w_k1_DIS_s114
                                 + 0.0170627026690046 w_k1_DIS_s115
                                 + 0.0433949772902841 w_k1_DIS_s116
                                 + 0.017112210119954 w_k1_DIS_s117
                                 + 0.0369442199889417 w_k1_FDX_s000
                                 + 0.0142461271987741 w_k1_FDX_s001
                                 + 0.00962604607586332 w_k1_FDX_s002
                                 + 0.0141206360213433 w_k1_FDX_s003
                                 + 0.0144689427167018 w_k1_FDX_s004
                                 + 0.0185740417956313 w_k1_FDX_s005
                                 + 0.0172038063828865 w_k1_FDX_s006
                                 + 0.0392998147997937 w_k1_FDX_s007
                                 + 0.0183242023903877 w_k1_FDX_s010
                                 + 0.0295795325721212 w_k1_FDX_s011
                                 + 0.0285341919951352 w_k1_FDX_s012
                                 + 0.0102769763383192 w_k1_FDX_s013
                                 + 0.0225543223107418 w_k1_FDX_s014
                                 + 0.020821925275222 w_k1_FDX_s015
                                 + 0.0430799194113043 w_k1_FDX_s016
                                 + 0.0223082793031649 w_k1_FDX_s017
                                 + 0.0314902524380926 w_k1_FDX_s100
                                 + 0.0552933084415262 w_k1_FDX_s101
                                 + 0.0578372721686111 w_k1_FDX_s102
                                 + 0.115438760480366 w_k1_FDX_s103
                                 + 0.0333186537245701 w_k1_FDX_s104
                                 + 0.024938646372761 w_k1_FDX_s105
                                 + 0.0366694884857166 w_k1_FDX_s106
                                 + 0.0703636069304497 w_k1_FDX_s107
                                 + 0.0406278648136221 w_k1_FDX_s110
                                 + 0.0203111758129366 w_k1_FDX_s111
                                 + 0.0592987786941231 w_k1_FDX_s112
                                 + 0.0215323233928986 w_k1_FDX_s113
                                 + 0.0129659318606832 w_k1_FDX_s114
                                 + 0.0163426607782451 w_k1_FDX_s115
                                 + 0.0442907208298359 w_k1_FDX_s116
                                 + 0.0176937412155046 w_k1_FDX_s117
                                 + 0.0398789379583503 w_k1_IBM_s000
                                 + 0.0151604729830314 w_k1_IBM_s001
                                 + 0.00991664542023378 w_k1_IBM_s002
                                 + 0.0134857526356477 w_k1_IBM_s003
                                 + 0.0141566176860818 w_k1_IBM_s004
                                 + 0.0187203417823159 w_k1_IBM_s005
                                 + 0.0174357541348798 w_k1_IBM_s006
                                 + 0.0408835560057489 w_k1_IBM_s007
                                 + 0.0179784101477568 w_k1_IBM_s010
                                 + 0.0307968887902793 w_k1_IBM_s011
                                 + 0.0296065118446746 w_k1_IBM_s012
                                 + 0.0101000118524024 w_k1_IBM_s013
                                 + 0.0238824336996224 w_k1_IBM_s014
                                 + 0.020519373230331 w_k1_IBM_s015
                                 + 0.0418797263472715 w_k1_IBM_s016
                                 + 0.0226896956725454 w_k1_IBM_s017
                                 + 0.0317645586718283 w_k1_IBM_s100
                                 + 0.0553758211002909 w_k1_IBM_s101
                                 + 0.0584752172623597 w_k1_IBM_s102
                                 + 0.113654846027058 w_k1_IBM_s103
                                 + 0.0343902310737479 w_k1_IBM_s104
                                 + 0.0249686746961183 w_k1_IBM_s105
                                 + 0.0366199145045269 w_k1_IBM_s106
                                 + 0.0717304069414386 w_k1_IBM_s107
                                 + 0.0412362122858969 w_k1_IBM_s110
                                 + 0.0199655291551311 w_k1_IBM_s111
                                 + 0.0594282230560187 w_k1_IBM_s112
                                 + 0.0212993137395643 w_k1_IBM_s113
                                 + 0.0136526847995509 w_k1_IBM_s114
                                 + 0.0173070389789582 w_k1_IBM_s115
                                 + 0.0447363279616466 w_k1_IBM_s116
                                 + 0.0168006081183088 w_k1_IBM_s117
                                 + 0.038133051846425 w_k1_INTC_s000
                                 + 0.0148338595603052 w_k1_INTC_s001
                                 + 0.00990742543805408 w_k1_INTC_s002
                                 + 0.0136798935351231 w_k1_INTC_s003
                                 + 0.0148505955541745 w_k1_INTC_s004
                                 + 0.0174391350010927 w_k1_INTC_s005
                                 + 0.019464561018038 w_k1_INTC_s006
                                 + 0.0440681623656701 w_k1_INTC_s007
                                 + 0.0194557383191234 w_k1_INTC_s010
                                 + 0.0320740974366376 w_k1_INTC_s011
                                 + 0.0310083870559587 w_k1_INTC_s012
                                 + 0.0102964584242399 w_k1_INTC_s013
                                 + 0.0219357641080501 w_k1_INTC_s014
                                 + 0.0229481906702882 w_k1_INTC_s015
                                 + 0.0376822150955158 w_k1_INTC_s016
                                 + 0.0194259101589844 w_k1_INTC_s017
                                 + 0.028456462995199 w_k1_INTC_s100
                                 + 0.0546676080516207 w_k1_INTC_s101
                                 + 0.0571865118428289 w_k1_INTC_s102
                                 + 0.112760721087304 w_k1_INTC_s103
                                 + 0.0293148692877346 w_k1_INTC_s104
                                 + 0.023146834750158 w_k1_INTC_s105
                                 + 0.0371214474717333 w_k1_INTC_s106
                                 + 0.0780454115246597 w_k1_INTC_s107
                                 + 0.0384546221532517 w_k1_INTC_s110
                                 + 0.0226861034204162 w_k1_INTC_s111
                                 + 0.0549215202494817 w_k1_INTC_s112
                                 + 0.0218587004927889 w_k1_INTC_s113
                                 + 0.012051113336943 w_k1_INTC_s114
                                 + 0.0152115235321535 w_k1_INTC_s115
                                 + 0.0447526178042491 w_k1_INTC_s116
                                 + 0.0159670052626373 w_k1_INTC_s117
                                 + 0.0370988080611644 w_k1_KO_s000
                                 + 0.0165503454992391 w_k1_KO_s001
                                 + 0.00969421140698475 w_k1_KO_s002
                                 + 0.0146209323280313 w_k1_KO_s003
                                 + 0.0150357907409847 w_k1_KO_s004
                                 + 0.0180645143612543 w_k1_KO_s005
                                 + 0.0173242291246613 w_k1_KO_s006
                                 + 0.042127360190534 w_k1_KO_s007
                                 + 0.0169605663566573 w_k1_KO_s010
                                 + 0.0297266144202099 w_k1_KO_s011
                                 + 0.0288207886441711 w_k1_KO_s012
                                 + 0.00866520021958532 w_k1_KO_s013
                                 + 0.0202468182659224 w_k1_KO_s014
                                 + 0.0211640401115289 w_k1_KO_s015
                                 + 0.041558664112057 w_k1_KO_s016
                                 + 0.021054660279222 w_k1_KO_s017
                                 + 0.0310447558636967 w_k1_KO_s100
                                 + 0.0598941184375506 w_k1_KO_s101
                                 + 0.0559268657652224 w_k1_KO_s102
                                 + 0.0980806599789215 w_k1_KO_s103
                                 + 0.0336684958467822 w_k1_KO_s104
                                 + 0.0227074433677628 w_k1_KO_s105
                                 + 0.0342955028110669 w_k1_KO_s106
                                 + 0.0692507972352418 w_k1_KO_s107
                                 + 0.0389774950771268 w_k1_KO_s110
                                 + 0.0222924191517699 w_k1_KO_s111
                                 + 0.0494332410643443 w_k1_KO_s112
                                 + 0.0224935493694249 w_k1_KO_s113
                                 + 0.0134329107701103 w_k1_KO_s114
                                 + 0.0157213243293795 w_k1_KO_s115
                                 + 0.0485289186925289 w_k1_KO_s116
                                 + 0.0160572416426358 w_k1_KO_s117
                                 + 0.0366641320193374 w_k1_MAT_s000
                                 + 0.0136112108994954 w_k1_MAT_s001
                                 + 0.00996120901708538 w_k1_MAT_s002
                                 + 0.0146826034315981 w_k1_MAT_s003
                                 + 0.0150904513900382 w_k1_MAT_s004
                                 + 0.0218425263326331 w_k1_MAT_s005
                                 + 0.0175667658528094 w_k1_MAT_s006
                                 + 0.0358222873869203 w_k1_MAT_s007
                                 + 0.0178833804008132 w_k1_MAT_s010
                                 + 0.0316390835807602 w_k1_MAT_s011
                                 + 0.0340152599765209 w_k1_MAT_s012
                                 + 0.00964627869602674 w_k1_MAT_s013
                                 + 0.0230773273616043 w_k1_MAT_s014
                                 + 0.0213085059710802 w_k1_MAT_s015
                                 + 0.0409478852273157 w_k1_MAT_s016
                                 + 0.0234331463361685 w_k1_MAT_s017
                                 + 0.034492505723339 w_k1_MAT_s100
                                 + 0.0556578890759631 w_k1_MAT_s101
                                 + 0.054252922032498 w_k1_MAT_s102
                                 + 0.125499246698637 w_k1_MAT_s103
                                 + 0.0390006998705348 w_k1_MAT_s104
                                 + 0.0196167559091583 w_k1_MAT_s105
                                 + 0.0377232733659841 w_k1_MAT_s106
                                 + 0.0688503963035007 w_k1_MAT_s107
                                 + 0.0341605645918818 w_k1_MAT_s110
                                 + 0.0149524585578856 w_k1_MAT_s111
                                 + 0.0635399327414882 w_k1_MAT_s112
                                 + 0.0275979435296544 w_k1_MAT_s113
                                 + 0.0139292140710358 w_k1_MAT_s114
                                 + 0.0146391001247969 w_k1_MAT_s115
                                 + 0.0415743460130136 w_k1_MAT_s116
                                 + 0.0198384918788688 w_k1_MAT_s117
                                 + 0.0362556302028443 w_k1_MSFT_s000
                                 + 0.0145757005001878 w_k1_MSFT_s001
                                 + 0.00854791354611247 w_k1_MSFT_s002
                                 + 0.0146768634252169 w_k1_MSFT_s003
                                 + 0.0142237005243561 w_k1_MSFT_s004
                                 + 0.0191617221334263 w_k1_MSFT_s005
                                 + 0.0204845096901845 w_k1_MSFT_s006
                                 + 0.0413155635111261 w_k1_MSFT_s007
                                 + 0.0182528877548979 w_k1_MSFT_s010
                                 + 0.0267922155194474 w_k1_MSFT_s011
                                 + 0.0319286480231272 w_k1_MSFT_s012
                                 + 0.010258788841726 w_k1_MSFT_s013
                                 + 0.0217243367838967 w_k1_MSFT_s014
                                 + 0.023537651163231 w_k1_MSFT_s015
                                 + 0.042354804306498 w_k1_MSFT_s016
                                 + 0.0207645771895691 w_k1_MSFT_s017
                                 + 0.0307901104717149 w_k1_MSFT_s100
                                 + 0.0527032747270976 w_k1_MSFT_s101
                                 + 0.0555444637392825 w_k1_MSFT_s102
                                 + 0.110642396740006 w_k1_MSFT_s103
                                 + 0.0332624841541098 w_k1_MSFT_s104
                                 + 0.0235710682043073 w_k1_MSFT_s105
                                 + 0.0371764258539266 w_k1_MSFT_s106
                                 + 0.0703406556174538 w_k1_MSFT_s107
                                 + 0.0361019367290094 w_k1_MSFT_s110
                                 + 0.0197174429367361 w_k1_MSFT_s111
                                 + 0.0560247925625552 w_k1_MSFT_s112
                                 + 0.0215789156105502 w_k1_MSFT_s113
                                 + 0.0126766041891309 w_k1_MSFT_s114
                                 + 0.0163631051904908 w_k1_MSFT_s115
                                 + 0.0435107377445758 w_k1_MSFT_s116
                                 + 0.0182069552521959 w_k1_MSFT_s117
                                 + 0.0382998490371997 w_k1_PG_s000
                                 + 0.0144311180873425 w_k1_PG_s001
                                 + 0.0100733377478381 w_k1_PG_s002
                                 + 0.0139958014400043 w_k1_PG_s003
                                 + 0.0142666937565343 w_k1_PG_s004
                                 + 0.0187751725306825 w_k1_PG_s005
                                 + 0.0167090406404354 w_k1_PG_s006
                                 + 0.0370852441956018 w_k1_PG_s007
                                 + 0.0199229701658934 w_k1_PG_s010
                                 + 0.027872913749289 w_k1_PG_s011
                                 + 0.0315252186124941 w_k1_PG_s012
                                 + 0.00953155810470749 w_k1_PG_s013
                                 + 0.021815899632565 w_k1_PG_s014
                                 + 0.0201279587825081 w_k1_PG_s015
                                 + 0.040401494943683 w_k1_PG_s016
                                 + 0.0207081638116364 w_k1_PG_s017
                                 + 0.0278576986075096 w_k1_PG_s100
                                 + 0.0510716317812793 w_k1_PG_s101
                                 + 0.0499055179377884 w_k1_PG_s102
                                 + 0.11623513327482 w_k1_PG_s103
                                 + 0.0316747795714396 w_k1_PG_s104
                                 + 0.0233864798044435 w_k1_PG_s105
                                 + 0.0358114574019031 w_k1_PG_s106
                                 + 0.0730061948113788 w_k1_PG_s107
                                 + 0.0359775776485498 w_k1_PG_s110
                                 + 0.0198991993931684 w_k1_PG_s111
                                 + 0.0533238961933256 w_k1_PG_s112
                                 + 0.0204942970625116 w_k1_PG_s113
                                 + 0.0135558245659803 w_k1_PG_s114
                                 + 0.0161032029753717 w_k1_PG_s115
                                 + 0.0429829432396445 w_k1_PG_s116
                                 + 0.0163482407058047 w_k1_PG_s117 >= 1.06
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
