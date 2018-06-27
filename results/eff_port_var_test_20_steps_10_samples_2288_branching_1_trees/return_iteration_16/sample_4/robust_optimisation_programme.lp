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
 w_k1_ADM_s0_return_constraint:  1.01044 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.04884 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.36755 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.989088 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.25597 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.92849 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.719045 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.41793 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.28823 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.95344 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.08911 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.872259 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.684573 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.30115 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.717595 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.08085 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.21937 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.15183 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.908917 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.946164 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.612508 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.760232 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.932368 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.682418 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.854573 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.0632 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.17045 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.2134 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.861122 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.54214 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.31986 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.996722 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.02912 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.796491 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.70578 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.17196 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.857043 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.17251 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.967866 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.07695 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.53401 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.93753 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.45809 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.859965 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.738276 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.58016 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.18011 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.951201 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.22579 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.897209 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.751642 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.46802 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.754619 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.28724 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.35168 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.970797 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.901366 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.24359 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.47842 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.715582 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.746472 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.633078 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.08233 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.979817 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.03404 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.38253 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.847586 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.36149 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.21074 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.993685 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.09282 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.950376 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.604731 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.59939 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.872569 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.43061 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.02931 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.08601 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.5956 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.878346 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.45906 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.81519 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.724412 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.99374 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.16161 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.932718 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.28541 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.1242 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.750108 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.4104 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.680626 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.33755 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.49734 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.00842 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.931699 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.15149 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.478071 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.62996 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.953437 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.747914 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.19507 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.29517 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.07337 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.23064 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.786779 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.76728 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.33598 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.12666 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.04928 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.797157 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.565758 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.63758 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.867791 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.56129 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.03849 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.994047 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.35121 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.895328 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.29866 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.908439 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.856032 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.53923 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.20414 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.04716 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.05958 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.99763 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.71185 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.43265 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.747597 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.17027 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.42719 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.16622 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.982493 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.07294 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.651208 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.826749 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.00829 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.778754 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.933969 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.10465 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.12499 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.18432 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.835431 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.42391 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.33716 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.0688 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.9416 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.954126 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.666932 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.1957 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.800315 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.30103 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.882911 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.26722 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.34767 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.951022 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.28672 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.901508 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.774851 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.40863 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.773641 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.763803 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.46947 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.13985 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.1807 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.06515 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.04367 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.7818 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.96647 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.696211 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.812962 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.23066 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.610253 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.614332 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.696451 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.756117 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.77029 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.10343 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.777651 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.19224 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.04855 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.14922 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.762887 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.990779 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.20008 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.743332 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.73635 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.66155 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.42387 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.09705 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.02235 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.00877 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.15622 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.999616 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.38783 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.796056 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.991935 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.61425 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.09932 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.11171 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.09869 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.16362 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.874065 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.48924 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.776639 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.14012 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.13101 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.06458 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.873634 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.01816 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.665419 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.750021 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.91152 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.856615 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.979537 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.13171 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.967034 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.20539 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.752057 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.19515 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.30562 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.21853 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.25365 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.23363 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.836217 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.32587 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.0944 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.29585 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.941048 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.15246 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.41072 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.958463 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.28946 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.892008 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.762734 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.61776 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.89988 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.892385 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.16085 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.19408 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.935697 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.09693 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.826089 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.35793 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.17202 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.906329 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.853969 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.14701 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.656852 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.66096 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.823466 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.749387 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.42009 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.21872 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.944616 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.16728 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.913791 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.24118 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.966771 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.894176 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.08043 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.814328 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.70557 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.36949 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.15873 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.11905 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.910797 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.36629 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.65155 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.96487 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.25044 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.887659 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.590357 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.10958 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.726527 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.761724 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.407 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.90151 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.915835 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.903152 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.934977 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.49564 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.06301 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.75127 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.902238 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.26465 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.536266 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.561656 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.612237 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.56501 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.83686 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.97593 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.925386 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.41404 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.23497 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.25769 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.87458 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.698104 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.04586 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.752044 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.564908 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.80483 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.18682 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.17625 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.00915 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.01946 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.33638 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.984854 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.29351 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.896025 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.85876 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.41144 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.02164 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.979343 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.970173 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.06594 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.805425 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.1939 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.772789 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.15514 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.29097 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.00998 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.838404 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.998444 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.729117 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.78345 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.989453 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.709183 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.03449 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.04681 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.06375 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.07366 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.773265 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.40995 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.21859 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.10683 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.09576 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.925107 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.770368 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.34743 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.909571 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.2973 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.982422 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.05053 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.31484 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.920717 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.19797 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.958491 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.83223 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.31619 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.06862 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.959926 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.07308 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.04437 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.809167 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.02512 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.855379 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.25829 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.20173 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.03411 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.953268 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.10727 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.744765 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.793502 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.938611 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.816835 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.21674 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.09437 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.996874 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.1702 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.924891 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.33926 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.995613 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.966859 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.966698 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.839925 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.778594 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.16336 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.917093 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.13402 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.391000853680443 z_k1_s0000
                                 - 0.284000077302538 z_k1_s0001
                                 - 0.104999882819633 z_k1_s0002
                                 - 0.357001121186416 z_k1_s0003
                                 - 0.39799887213552 z_k1_s0004
                                 - 0.313000850900164 z_k1_s0005
                                 - 0.339999944939691 z_k1_s0006
                                 - 0.431998604629547 z_k1_s0007
                                 - 0.0609999429328612 z_k1_s0010
                                 - 0.0539999470422059 z_k1_s0011
                                 - 0.0639999730574164 z_k1_s0012
                                 - 0.221999848417934 z_k1_s0013
                                 - 0.047999961193069 z_k1_s0014
                                 - 0.120999652624284 z_k1_s0015
                                 - 0.0920000310200111 z_k1_s0016
                                 - 0.0819999306048273 z_k1_s0017
                                 - 0.157000213422891 z_k1_s0020
                                 - 0.144999463690742 z_k1_s0021
                                 - 0.135999665640785 z_k1_s0022
                                 - 0.0919999056862676 z_k1_s0023
                                 - 0.0639999565425964 z_k1_s0024
                                 - 0.084999892925378 z_k1_s0025
                                 - 0.120000364329375 z_k1_s0026
                                 - 0.21999981877146 z_k1_s0027
                                 - 0.256001067990679 z_k1_s0030
                                 - 0.34100022657868 z_k1_s0031
                                 - 0.271999953487476 z_k1_s0032
                                 - 0.290000353425041 z_k1_s0033
                                 - 0.136000354744576 z_k1_s0034
                                 - 0.167000392524495 z_k1_s0035
                                 - 0.32100020220454 z_k1_s0036
                                 - 0.107000319402072 z_k1_s0037
                                 - 0.449998170873585 z_k1_s0040
                                 - 0.227000090714345 z_k1_s0041
                                 - 0.242000171809867 z_k1_s0042
                                 - 0.28099953025981 z_k1_s0043
                                 - 0.219000331596205 z_k1_s0044
                                 - 0.174999738550002 z_k1_s0045
                                 - 0.197999578763664 z_k1_s0046
                                 - 0.338998849364344 z_k1_s0047
                                 - 0.181000347270775 z_k1_s0050
                                 - 0.306000680712075 z_k1_s0051
                                 - 0.257000668611284 z_k1_s0052
                                 - 0.390000701456287 z_k1_s0053
                                 - 0.203000261932202 z_k1_s0054
                                 - 0.49900163609018 z_k1_s0055
                                 - 0.113000148934679 z_k1_s0056
                                 - 0.169000374564509 z_k1_s0057
                                 - 0.285000024827914 z_k1_s0060
                                 - 0.394999562989569 z_k1_s0061
                                 - 0.188000056267185 z_k1_s0062
                                 - 0.115000024523544 z_k1_s0063
                                 - 0.334000326454465 z_k1_s0064
                                 - 0.1779999103434 z_k1_s0065
                                 - 0.292000953774557 z_k1_s0066
                                 - 0.279999951866021 z_k1_s0067
                                 - 0.205999970846777 z_k1_s0070
                                 - 0.0630000104939083 z_k1_s0071
                                 - 0.129999585320967 z_k1_s0072
                                 - 0.0870001453915265 z_k1_s0073
                                 - 0.110999697189417 z_k1_s0074
                                 - 0.0689999400933614 z_k1_s0075
                                 - 0.0799999498589607 z_k1_s0076
                                 - 0.0870001453915265 z_k1_s0077
                                 - 0.712001254747982 z_k1_s0100
                                 - 0.640001034121612 z_k1_s0101
                                 - 1.26800481247721 z_k1_s0102
                                 - 1.29100650520802 z_k1_s0103
                                 - 1.47300103739152 z_k1_s0104
                                 - 0.627000947936211 z_k1_s0105
                                 - 1.33800270947086 z_k1_s0106
                                 - 0.994003509370456 z_k1_s0107
                                 - 0.653000531005792 z_k1_s0110
                                 - 0.669002129253116 z_k1_s0111
                                 - 0.358000918127759 z_k1_s0112
                                 - 1.25000414266614 z_k1_s0113
                                 - 0.49000109188789 z_k1_s0114
                                 - 0.958002443966432 z_k1_s0115
                                 - 0.251000711560682 z_k1_s0116
                                 - 1.15400033619566 z_k1_s0117
                                 - 1.9759973566744 z_k1_s0120
                                 - 0.792002062628758 z_k1_s0121
                                 - 0.410999323363692 z_k1_s0122
                                 - 0.620999140009066 z_k1_s0123
                                 - 0.46699878013673 z_k1_s0124
                                 - 2.00899404408205 z_k1_s0125
                                 - 0.425999098410192 z_k1_s0126
                                 - 0.712998353493141 z_k1_s0127
                                 - 0.926998468808138 z_k1_s0130
                                 - 0.93899793481877 z_k1_s0131
                                 - 0.408999917523026 z_k1_s0132
                                 - 0.362999734616344 z_k1_s0133
                                 - 0.212000043118019 z_k1_s0134
                                 - 0.228999983466183 z_k1_s0135
                                 - 0.272999976637853 z_k1_s0136
                                 - 0.829000042388901 z_k1_s0137
                                 - 1.08799513773251 z_k1_s0140
                                 - 1.64099998532084 z_k1_s0141
                                 - 0.508999438687051 z_k1_s0142
                                 - 0.897997275626871 z_k1_s0143
                                 - 0.63099954193198 z_k1_s0144
                                 - 0.995996172923462 z_k1_s0145
                                 - 0.581999355284608 z_k1_s0146
                                 - 1.03499942408789 z_k1_s0147
                                 - 1.70500391852413 z_k1_s0150
                                 - 1.98800179973869 z_k1_s0151
                                 - 1.11099783914204 z_k1_s0152
                                 - 0.551999971951647 z_k1_s0153
                                 - 1.16400197737608 z_k1_s0154
                                 - 0.661000387294227 z_k1_s0155
                                 - 1.14900381100763 z_k1_s0156
                                 - 1.03800239108272 z_k1_s0157
                                 - 0.176000166944455 z_k1_s0160
                                 - 0.414001291180575 z_k1_s0161
                                 - 0.858999833807457 z_k1_s0162
                                 - 0.594998776590893 z_k1_s0163
                                 - 0.338999963998748 z_k1_s0164
                                 - 0.787001012384543 z_k1_s0165
                                 - 0.25199990897961 z_k1_s0166
                                 - 0.611998050093592 z_k1_s0167
                                 - 0.252999350289109 z_k1_s0170
                                 - 0.209000179665636 z_k1_s0171
                                 - 0.323999253650961 z_k1_s0172
                                 - 0.15100005096983 z_k1_s0173
                                 - 0.373000438901948 z_k1_s0174
                                 - 0.213000143685991 z_k1_s0175
                                 - 0.309001014576263 z_k1_s0176
                                 - 0.336001313714206 z_k1_s0177
                                 - 0.183000131805217 z_k1_s1000
                                 - 0.0880001738515961 z_k1_s1001
                                 - 0.0439999425259197 z_k1_s1002
                                 - 0.0920000504813247 z_k1_s1003
                                 - 0.0479999635555267 z_k1_s1004
                                 - 0.0910000813238926 z_k1_s1005
                                 - 0.062999934116644 z_k1_s1006
                                 - 0.113000124786792 z_k1_s1007
                                 - 0.287998400377372 z_k1_s1010
                                 - 0.536998665638568 z_k1_s1011
                                 - 0.199999681760826 z_k1_s1012
                                 - 0.1699998591965 z_k1_s1013
                                 - 0.176999757268271 z_k1_s1014
                                 - 0.357999975091029 z_k1_s1015
                                 - 0.304999449835362 z_k1_s1016
                                 - 0.558998345292705 z_k1_s1017
                                 - 0.120999558569821 z_k1_s1020
                                 - 0.106999907363294 z_k1_s1021
                                 - 0.123000257884783 z_k1_s1022
                                 - 0.289000161046512 z_k1_s1023
                                 - 0.167999462477081 z_k1_s1024
                                 - 0.123000257884783 z_k1_s1025
                                 - 0.110999937993686 z_k1_s1026
                                 - 0.097999952444873 z_k1_s1027
                                 - 0.133000431499064 z_k1_s1030
                                 - 0.153000438949115 z_k1_s1031
                                 - 0.195000345463872 z_k1_s1032
                                 - 0.0700002599259284 z_k1_s1033
                                 - 0.271000280233766 z_k1_s1034
                                 - 0.186000209595924 z_k1_s1035
                                 - 0.267001432407456 z_k1_s1036
                                 - 0.284000822933024 z_k1_s1037
                                 - 0.307001299700034 z_k1_s1040
                                 - 0.15300045061875 z_k1_s1041
                                 - 0.115000322679413 z_k1_s1042
                                 - 0.216000275696172 z_k1_s1043
                                 - 0.159000543388298 z_k1_s1044
                                 - 0.175000790773758 z_k1_s1045
                                 - 0.237000983390531 z_k1_s1046
                                 - 0.170000330464859 z_k1_s1047
                                 - 0.163000477860607 z_k1_s1050
                                 - 0.0900000751953611 z_k1_s1051
                                 - 0.121999663558003 z_k1_s1052
                                 - 0.262999916744116 z_k1_s1053
                                 - 0.0979996821859205 z_k1_s1054
                                 - 0.086000078300014 z_k1_s1055
                                 - 0.0620000002278974 z_k1_s1056
                                 - 0.0830000564534952 z_k1_s1057
                                 - 0.557997647132174 z_k1_s1060
                                 - 0.335999701108896 z_k1_s1061
                                 - 0.272999895400613 z_k1_s1062
                                 - 0.323999632783787 z_k1_s1063
                                 - 0.206999519612512 z_k1_s1064
                                 - 0.127999696537223 z_k1_s1065
                                 - 0.222999979378352 z_k1_s1066
                                 - 0.162999637752806 z_k1_s1067
                                 - 0.0849998456091383 z_k1_s1070
                                 - 0.0949997579868258 z_k1_s1071
                                 - 0.167000202943979 z_k1_s1072
                                 - 0.244999755649459 z_k1_s1073
                                 - 0.19099925793193 z_k1_s1074
                                 - 0.160999127199671 z_k1_s1075
                                 - 0.178999730437954 z_k1_s1076
                                 - 0.188999774015374 z_k1_s1077
                                 - 0.139000125826693 z_k1_s1100
                                 - 0.422999458198951 z_k1_s1101
                                 - 0.342999251277075 z_k1_s1102
                                 - 0.14400013875931 z_k1_s1103
                                 - 0.232000132673307 z_k1_s1104
                                 - 0.223000109394595 z_k1_s1105
                                 - 0.225000114567642 z_k1_s1106
                                 - 0.608001105206124 z_k1_s1107
                                 - 0.197000044174848 z_k1_s1110
                                 - 0.610000040632779 z_k1_s1111
                                 - 0.349000932640707 z_k1_s1112
                                 - 0.324998713009288 z_k1_s1113
                                 - 0.371999226287543 z_k1_s1114
                                 - 0.152999943650518 z_k1_s1115
                                 - 0.302998933347119 z_k1_s1116
                                 - 0.396999959903629 z_k1_s1117
                                 - 0.323998203178923 z_k1_s1120
                                 - 0.121999539951934 z_k1_s1121
                                 - 0.1509995492582 z_k1_s1122
                                 - 0.538997552175724 z_k1_s1123
                                 - 0.589997501342108 z_k1_s1124
                                 - 0.30299855163888 z_k1_s1125
                                 - 0.474999144433115 z_k1_s1126
                                 - 0.279999993853875 z_k1_s1127
                                 - 0.429000695574543 z_k1_s1130
                                 - 1.43200005117827 z_k1_s1131
                                 - 0.787002692871691 z_k1_s1132
                                 - 0.354000673096605 z_k1_s1133
                                 - 0.900000269735258 z_k1_s1134
                                 - 0.612003585424397 z_k1_s1135
                                 - 0.577002629933464 z_k1_s1136
                                 - 0.578998472527472 z_k1_s1137
                                 - 0.415001008244441 z_k1_s1140
                                 - 0.453001100565619 z_k1_s1141
                                 - 0.306001109729688 z_k1_s1142
                                 - 0.332001172896809 z_k1_s1143
                                 - 0.807001960610275 z_k1_s1144
                                 - 0.655001591325564 z_k1_s1145
                                 - 0.437001061693544 z_k1_s1146
                                 - 0.258000993113463 z_k1_s1147
                                 - 0.468001684633793 z_k1_s1150
                                 - 0.283999419032652 z_k1_s1151
                                 - 0.359000586609437 z_k1_s1152
                                 - 0.535000226573334 z_k1_s1153
                                 - 0.429999120366339 z_k1_s1154
                                 - 0.112000035086368 z_k1_s1155
                                 - 0.14100023996249 z_k1_s1156
                                 - 0.313000680709782 z_k1_s1157
                                 - 0.471998693585969 z_k1_s1160
                                 - 0.170000111808779 z_k1_s1161
                                 - 0.314999990280872 z_k1_s1162
                                 - 0.738000130463829 z_k1_s1163
                                 - 0.561001039369287 z_k1_s1164
                                 - 0.619000722597998 z_k1_s1165
                                 - 0.18600022092024 z_k1_s1166
                                 - 0.290999994213759 z_k1_s1167
                                 - 0.242999278367854 z_k1_s1170
                                 - 0.173999530184271 z_k1_s1171
                                 - 0.423000261651334 z_k1_s1172
                                 - 0.237999288748774 z_k1_s1173
                                 - 0.399999169526432 z_k1_s1174
                                 - 0.161999609378342 z_k1_s1175
                                 - 0.633999660736937 z_k1_s1176
                                 - 0.439998000881807 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.775313 w_k1_ADM_s000
                                 + 0.65228 w_k1_AXP_s000
                                 + 0.708088 w_k1_DIS_s000
                                 + 0.767547 w_k1_FDX_s000
                                 + 0.67714 w_k1_IBM_s000
                                 + 0.817518 w_k1_INTC_s000
                                 + 0.716058 w_k1_KO_s000
                                 + 0.591015 w_k1_MAT_s000
                                 + 0.834522 w_k1_MSFT_s000
                                 + 0.851651 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.15392 w_k1_ADM_s000
                                 + 1.15005 w_k1_AXP_s000
                                 + 1.36197 w_k1_DIS_s000
                                 + 1.15694 w_k1_FDX_s000
                                 + 0.967009 w_k1_IBM_s000
                                 + 1.10527 w_k1_INTC_s000
                                 + 1.15998 w_k1_KO_s000 + 1.32198 w_k1_MAT_s000
                                 + 1.24821 w_k1_MSFT_s000 + 1.1394 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.37442 w_k1_ADM_s000
                                 + 1.67986 w_k1_AXP_s000
                                 + 1.57493 w_k1_DIS_s000
                                 + 1.42662 w_k1_FDX_s000
                                 + 0.919541 w_k1_IBM_s000
                                 + 1.0592 w_k1_INTC_s000 + 1.34336 w_k1_KO_s000
                                 + 1.91606 w_k1_MAT_s000 + 1.185 w_k1_MSFT_s000
                                 + 1.30975 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.02375 w_k1_ADM_s000
                                 + 1.16677 w_k1_AXP_s000
                                 + 0.859478 w_k1_DIS_s000
                                 + 0.935897 w_k1_FDX_s000
                                 + 1.26574 w_k1_IBM_s000
                                 + 0.952295 w_k1_INTC_s000
                                 + 1.08983 w_k1_KO_s000 + 1.6896 w_k1_MAT_s000
                                 + 0.955154 w_k1_MSFT_s000
                                 + 0.994001 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.0789 w_k1_ADM_s000
                                 + 1.12186 w_k1_AXP_s000
                                 + 0.966533 w_k1_DIS_s000
                                 + 1.02749 w_k1_FDX_s000
                                 + 0.797127 w_k1_IBM_s000
                                 + 0.968655 w_k1_INTC_s000
                                 + 1.02905 w_k1_KO_s000 + 1.18828 w_k1_MAT_s000
                                 + 0.975567 w_k1_MSFT_s000
                                 + 1.01053 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.09642 w_k1_ADM_s000
                                 + 0.733928 w_k1_AXP_s000
                                 + 1.16512 w_k1_DIS_s000
                                 + 0.937005 w_k1_FDX_s000
                                 + 0.79587 w_k1_IBM_s000
                                 + 0.908312 w_k1_INTC_s000
                                 + 0.963159 w_k1_KO_s000
                                 + 0.636524 w_k1_MAT_s000
                                 + 1.062 w_k1_MSFT_s000 + 0.961722 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.06104 w_k1_ADM_s000
                                 + 0.713326 w_k1_AXP_s000
                                 + 0.750709 w_k1_DIS_s000
                                 + 0.763194 w_k1_FDX_s000
                                 + 0.990611 w_k1_IBM_s000
                                 + 0.793587 w_k1_INTC_s000
                                 + 0.948121 w_k1_KO_s000
                                 + 1.18922 w_k1_MAT_s000
                                 + 0.9046 w_k1_MSFT_s000
                                 + 0.915728 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.1797 w_k1_ADM_s000
                                 + 1.38418 w_k1_AXP_s000
                                 + 1.38676 w_k1_DIS_s000
                                 + 1.10974 w_k1_FDX_s000
                                 + 1.46095 w_k1_IBM_s000
                                 + 0.974578 w_k1_INTC_s000
                                 + 1.37368 w_k1_KO_s000 + 2.09644 w_k1_MAT_s000
                                 + 1.11832 w_k1_MSFT_s000
                                 + 1.28906 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.21893 w_k1_ADM_s001
                                 + 1.79156 w_k1_AXP_s001
                                 + 1.20909 w_k1_DIS_s001
                                 + 1.32035 w_k1_FDX_s001
                                 + 0.922769 w_k1_IBM_s001
                                 + 1.10848 w_k1_INTC_s001
                                 + 1.03167 w_k1_KO_s001 + 1.83772 w_k1_MAT_s001
                                 + 1.19862 w_k1_MSFT_s001
                                 + 0.980461 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.15886 w_k1_ADM_s001
                                 + 1.58462 w_k1_AXP_s001
                                 + 1.70349 w_k1_DIS_s001
                                 + 1.50045 w_k1_FDX_s001
                                 + 0.835893 w_k1_IBM_s001
                                 + 1.02518 w_k1_INTC_s001
                                 + 0.965905 w_k1_KO_s001
                                 + 1.47602 w_k1_MAT_s001
                                 + 1.2546 w_k1_MSFT_s001 + 1.05908 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.923301 w_k1_ADM_s001
                                 + 0.928836 w_k1_AXP_s001
                                 + 0.696971 w_k1_DIS_s001
                                 + 0.712121 w_k1_FDX_s001
                                 + 1.46541 w_k1_IBM_s001
                                 + 0.727115 w_k1_INTC_s001
                                 + 1.0139 w_k1_KO_s001 + 1.98229 w_k1_MAT_s001
                                 + 0.871277 w_k1_MSFT_s001
                                 + 1.07199 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.00455 w_k1_ADM_s001
                                 + 0.920074 w_k1_AXP_s001
                                 + 0.901935 w_k1_DIS_s001
                                 + 0.95926 w_k1_FDX_s001
                                 + 0.808332 w_k1_IBM_s001
                                 + 0.842738 w_k1_INTC_s001
                                 + 0.744512 w_k1_KO_s001
                                 + 1.14058 w_k1_MAT_s001
                                 + 0.986178 w_k1_MSFT_s001
                                 + 0.811515 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.894356 w_k1_ADM_s001
                                 + 1.03353 w_k1_AXP_s001
                                 + 1.50892 w_k1_DIS_s001
                                 + 1.19792 w_k1_FDX_s001
                                 + 1.32958 w_k1_IBM_s001
                                 + 1.18967 w_k1_INTC_s001
                                 + 1.37135 w_k1_KO_s001 + 1.60318 w_k1_MAT_s001
                                 + 1.39821 w_k1_MSFT_s001
                                 + 1.13239 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.59357 w_k1_ADM_s001
                                 + 0.801871 w_k1_AXP_s001
                                 + 0.714075 w_k1_DIS_s001
                                 + 0.815488 w_k1_FDX_s001
                                 + 0.834837 w_k1_IBM_s001
                                 + 0.689011 w_k1_INTC_s001
                                 + 0.727877 w_k1_KO_s001
                                 + 1.06796 w_k1_MAT_s001
                                 + 0.844459 w_k1_MSFT_s001
                                 + 0.863517 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.05741 w_k1_ADM_s001
                                 + 1.00362 w_k1_AXP_s001 + 1.1248 w_k1_DIS_s001
                                 + 0.895683 w_k1_FDX_s001
                                 + 1.48542 w_k1_IBM_s001
                                 + 0.755356 w_k1_INTC_s001
                                 + 1.29375 w_k1_KO_s001 + 1.85505 w_k1_MAT_s001
                                 + 1.10979 w_k1_MSFT_s001
                                 + 1.02368 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.924914 w_k1_ADM_s001
                                 + 1.51083 w_k1_AXP_s001
                                 + 1.07423 w_k1_DIS_s001
                                 + 1.12019 w_k1_FDX_s001
                                 + 1.57871 w_k1_IBM_s001
                                 + 0.979081 w_k1_INTC_s001
                                 + 1.09855 w_k1_KO_s001 + 2.50482 w_k1_MAT_s001
                                 + 0.954842 w_k1_MSFT_s001
                                 + 1.01233 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.798971 w_k1_ADM_s002
                                 + 0.803271 w_k1_AXP_s002
                                 + 0.928137 w_k1_DIS_s002
                                 + 1.02829 w_k1_FDX_s002
                                 + 0.804607 w_k1_IBM_s002
                                 + 1.05611 w_k1_INTC_s002
                                 + 0.853704 w_k1_KO_s002 + 0.7688 w_k1_MAT_s002
                                 + 1.05006 w_k1_MSFT_s002
                                 + 0.738482 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.07616 w_k1_ADM_s002
                                 + 1.2131 w_k1_AXP_s002 + 1.53748 w_k1_DIS_s002
                                 + 1.32542 w_k1_FDX_s002
                                 + 1.24363 w_k1_IBM_s002
                                 + 1.34768 w_k1_INTC_s002
                                 + 1.32213 w_k1_KO_s002 + 1.12046 w_k1_MAT_s002
                                 + 1.4734 w_k1_MSFT_s002 + 1.05789 w_k1_PG_s002
                                 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.834081 w_k1_ADM_s002
                                 + 1.17178 w_k1_AXP_s002
                                 + 1.08269 w_k1_DIS_s002 + 1.1972 w_k1_FDX_s002
                                 + 0.993056 w_k1_IBM_s002
                                 + 1.09346 w_k1_INTC_s002
                                 + 1.03266 w_k1_KO_s002 + 1.38576 w_k1_MAT_s002
                                 + 1.27822 w_k1_MSFT_s002
                                 + 1.02261 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.90151 w_k1_ADM_s002
                                 + 1.18081 w_k1_AXP_s002
                                 + 0.969503 w_k1_DIS_s002
                                 + 0.99488 w_k1_FDX_s002
                                 + 1.09385 w_k1_IBM_s002
                                 + 0.965143 w_k1_INTC_s002
                                 + 0.953598 w_k1_KO_s002
                                 + 1.24253 w_k1_MAT_s002
                                 + 1.01574 w_k1_MSFT_s002
                                 + 0.923615 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.845141 w_k1_ADM_s002
                                 + 1.10045 w_k1_AXP_s002
                                 + 1.07436 w_k1_DIS_s002 + 0.9462 w_k1_FDX_s002
                                 + 1.53466 w_k1_IBM_s002
                                 + 1.13016 w_k1_INTC_s002
                                 + 1.32248 w_k1_KO_s002 + 1.86207 w_k1_MAT_s002
                                 + 1.26911 w_k1_MSFT_s002
                                 + 1.05754 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.01276 w_k1_ADM_s002
                                 + 0.986044 w_k1_AXP_s002
                                 + 1.7635 w_k1_DIS_s002 + 1.21243 w_k1_FDX_s002
                                 + 1.24972 w_k1_IBM_s002
                                 + 1.2425 w_k1_INTC_s002 + 1.39014 w_k1_KO_s002
                                 + 0.875966 w_k1_MAT_s002
                                 + 1.38312 w_k1_MSFT_s002
                                 + 1.02629 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.815167 w_k1_ADM_s002
                                 + 0.983729 w_k1_AXP_s002
                                 + 0.772174 w_k1_DIS_s002
                                 + 0.85649 w_k1_FDX_s002
                                 + 1.16393 w_k1_IBM_s002
                                 + 0.950004 w_k1_INTC_s002
                                 + 0.996889 w_k1_KO_s002
                                 + 0.841674 w_k1_MAT_s002
                                 + 0.854889 w_k1_MSFT_s002
                                 + 0.79954 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.880809 w_k1_ADM_s002
                                 + 0.768909 w_k1_AXP_s002
                                 + 0.730245 w_k1_DIS_s002
                                 + 0.811229 w_k1_FDX_s002
                                 + 0.903003 w_k1_IBM_s002
                                 + 0.810493 w_k1_INTC_s002
                                 + 0.915904 w_k1_KO_s002
                                 + 1.19044 w_k1_MAT_s002
                                 + 0.932643 w_k1_MSFT_s002
                                 + 0.862862 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.771036 w_k1_ADM_s003
                                 + 0.64858 w_k1_AXP_s003
                                 + 0.723684 w_k1_DIS_s003
                                 + 0.786322 w_k1_FDX_s003
                                 + 0.866971 w_k1_IBM_s003
                                 + 0.73114 w_k1_INTC_s003
                                 + 0.862059 w_k1_KO_s003
                                 + 0.749205 w_k1_MAT_s003
                                 + 0.891005 w_k1_MSFT_s003
                                 + 0.79507 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.16055 w_k1_ADM_s003
                                 + 1.26183 w_k1_AXP_s003
                                 + 1.37673 w_k1_DIS_s003
                                 + 1.27088 w_k1_FDX_s003
                                 + 1.24649 w_k1_IBM_s003
                                 + 1.20159 w_k1_INTC_s003 + 1.176 w_k1_KO_s003
                                 + 0.904398 w_k1_MAT_s003
                                 + 1.19519 w_k1_MSFT_s003
                                 + 1.00484 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.745489 w_k1_ADM_s003
                                 + 0.759482 w_k1_AXP_s003
                                 + 0.724475 w_k1_DIS_s003
                                 + 0.863848 w_k1_FDX_s003
                                 + 0.596804 w_k1_IBM_s003
                                 + 0.906989 w_k1_INTC_s003
                                 + 0.663163 w_k1_KO_s003
                                 + 0.443111 w_k1_MAT_s003
                                 + 0.884663 w_k1_MSFT_s003
                                 + 0.763752 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.992776 w_k1_ADM_s003
                                 + 0.970002 w_k1_AXP_s003
                                 + 1.19679 w_k1_DIS_s003
                                 + 1.11122 w_k1_FDX_s003
                                 + 0.644989 w_k1_IBM_s003
                                 + 1.32596 w_k1_INTC_s003
                                 + 0.938757 w_k1_KO_s003
                                 + 0.544488 w_k1_MAT_s003
                                 + 1.28687 w_k1_MSFT_s003
                                 + 0.927206 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.949286 w_k1_ADM_s003
                                 + 1.36811 w_k1_AXP_s003
                                 + 1.09244 w_k1_DIS_s003
                                 + 0.996276 w_k1_FDX_s003
                                 + 1.65536 w_k1_IBM_s003
                                 + 1.10435 w_k1_INTC_s003
                                 + 1.11326 w_k1_KO_s003 + 1.54248 w_k1_MAT_s003
                                 + 1.05977 w_k1_MSFT_s003
                                 + 0.952614 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.961695 w_k1_ADM_s003
                                 + 0.828437 w_k1_AXP_s003
                                 + 1.06268 w_k1_DIS_s003
                                 + 0.943292 w_k1_FDX_s003
                                 + 1.08572 w_k1_IBM_s003
                                 + 1.02647 w_k1_INTC_s003
                                 + 1.05517 w_k1_KO_s003
                                 + 0.657885 w_k1_MAT_s003
                                 + 1.07194 w_k1_MSFT_s003
                                 + 0.922439 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.01974 w_k1_ADM_s003
                                 + 0.950719 w_k1_AXP_s003
                                 + 0.997178 w_k1_DIS_s003
                                 + 0.997472 w_k1_FDX_s003
                                 + 1.01831 w_k1_IBM_s003
                                 + 0.892146 w_k1_INTC_s003
                                 + 0.967458 w_k1_KO_s003
                                 + 0.732619 w_k1_MAT_s003
                                 + 1.03944 w_k1_MSFT_s003
                                 + 0.945406 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.941031 w_k1_ADM_s003
                                 + 1.4181 w_k1_AXP_s003
                                 + 0.868161 w_k1_DIS_s003
                                 + 1.02328 w_k1_FDX_s003
                                 + 0.938443 w_k1_IBM_s003
                                 + 0.893889 w_k1_INTC_s003
                                 + 0.880084 w_k1_KO_s003
                                 + 1.35911 w_k1_MAT_s003
                                 + 0.938212 w_k1_MSFT_s003
                                 + 0.945686 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.13115 w_k1_ADM_s004
                                 + 0.994863 w_k1_AXP_s004
                                 + 1.18386 w_k1_DIS_s004
                                 + 1.13002 w_k1_FDX_s004
                                 + 0.875742 w_k1_IBM_s004
                                 + 1.04426 w_k1_INTC_s004
                                 + 1.03798 w_k1_KO_s004
                                 + 0.859067 w_k1_MAT_s004
                                 + 1.09089 w_k1_MSFT_s004
                                 + 1.08712 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.03573 w_k1_ADM_s004
                                 + 1.54333 w_k1_AXP_s004
                                 + 1.52878 w_k1_DIS_s004
                                 + 1.41579 w_k1_FDX_s004
                                 + 1.00224 w_k1_IBM_s004
                                 + 1.21796 w_k1_INTC_s004
                                 + 1.19612 w_k1_KO_s004 + 1.25346 w_k1_MAT_s004
                                 + 1.41491 w_k1_MSFT_s004
                                 + 1.12738 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.998636 w_k1_ADM_s004
                                 + 1.01527 w_k1_AXP_s004
                                 + 0.871588 w_k1_DIS_s004
                                 + 1.01476 w_k1_FDX_s004
                                 + 0.782192 w_k1_IBM_s004
                                 + 1.01089 w_k1_INTC_s004
                                 + 0.972701 w_k1_KO_s004
                                 + 0.745345 w_k1_MAT_s004
                                 + 1.01901 w_k1_MSFT_s004
                                 + 0.893142 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.946044 w_k1_ADM_s004
                                 + 1.2115 w_k1_AXP_s004 + 1.23824 w_k1_DIS_s004
                                 + 1.17201 w_k1_FDX_s004
                                 + 1.57378 w_k1_IBM_s004
                                 + 1.17385 w_k1_INTC_s004
                                 + 1.31903 w_k1_KO_s004 + 1.4773 w_k1_MAT_s004
                                 + 1.17818 w_k1_MSFT_s004
                                 + 1.17585 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.971273 w_k1_ADM_s004
                                 + 0.753676 w_k1_AXP_s004
                                 + 0.69843 w_k1_DIS_s004
                                 + 0.756919 w_k1_FDX_s004
                                 + 0.8508 w_k1_IBM_s004
                                 + 0.728842 w_k1_INTC_s004
                                 + 0.806993 w_k1_KO_s004
                                 + 0.935438 w_k1_MAT_s004
                                 + 0.833736 w_k1_MSFT_s004
                                 + 0.872864 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.87939 w_k1_ADM_s004
                                 + 0.798044 w_k1_AXP_s004
                                 + 0.784553 w_k1_DIS_s004
                                 + 0.86079 w_k1_FDX_s004
                                 + 0.716086 w_k1_IBM_s004
                                 + 0.921291 w_k1_INTC_s004
                                 + 0.852628 w_k1_KO_s004
                                 + 0.66759 w_k1_MAT_s004
                                 + 0.984038 w_k1_MSFT_s004
                                 + 0.872186 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.799922 w_k1_ADM_s004
                                 + 1.22643 w_k1_AXP_s004
                                 + 0.755061 w_k1_DIS_s004
                                 + 1.03961 w_k1_FDX_s004
                                 + 0.745939 w_k1_IBM_s004
                                 + 1.2114 w_k1_INTC_s004
                                 + 0.921829 w_k1_KO_s004
                                 + 0.971785 w_k1_MAT_s004
                                 + 1.03577 w_k1_MSFT_s004
                                 + 0.947084 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.609432 w_k1_ADM_s004
                                 + 1.09561 w_k1_AXP_s004
                                 + 0.83758 w_k1_DIS_s004
                                 + 0.899697 w_k1_FDX_s004
                                 + 1.23031 w_k1_IBM_s004
                                 + 1.06614 w_k1_INTC_s004
                                 + 1.23384 w_k1_KO_s004 + 1.12687 w_k1_MAT_s004
                                 + 1.03127 w_k1_MSFT_s004
                                 + 0.957606 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.10625 w_k1_ADM_s005
                                 + 1.10603 w_k1_AXP_s005
                                 + 1.30943 w_k1_DIS_s005
                                 + 1.26719 w_k1_FDX_s005
                                 + 0.69263 w_k1_IBM_s005
                                 + 1.02646 w_k1_INTC_s005
                                 + 0.899639 w_k1_KO_s005
                                 + 0.912446 w_k1_MAT_s005
                                 + 1.03355 w_k1_MSFT_s005
                                 + 0.926778 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.776066 w_k1_ADM_s005
                                 + 0.806277 w_k1_AXP_s005
                                 + 0.718951 w_k1_DIS_s005
                                 + 0.843361 w_k1_FDX_s005
                                 + 0.661402 w_k1_IBM_s005
                                 + 0.904654 w_k1_INTC_s005
                                 + 0.684013 w_k1_KO_s005
                                 + 0.819939 w_k1_MAT_s005
                                 + 0.900939 w_k1_MSFT_s005
                                 + 0.825695 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.42712 w_k1_ADM_s005
                                 + 1.6829 w_k1_AXP_s005 + 1.47388 w_k1_DIS_s005
                                 + 1.34019 w_k1_FDX_s005
                                 + 1.25494 w_k1_IBM_s005
                                 + 1.19521 w_k1_INTC_s005
                                 + 1.24727 w_k1_KO_s005 + 1.29494 w_k1_MAT_s005
                                 + 1.39652 w_k1_MSFT_s005 + 1.2228 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.10444 w_k1_ADM_s005
                                 + 1.18731 w_k1_AXP_s005
                                 + 1.12941 w_k1_DIS_s005
                                 + 1.10709 w_k1_FDX_s005
                                 + 0.978174 w_k1_IBM_s005
                                 + 1.13336 w_k1_INTC_s005
                                 + 0.960757 w_k1_KO_s005
                                 + 1.09245 w_k1_MAT_s005
                                 + 1.10468 w_k1_MSFT_s005
                                 + 1.03832 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.789019 w_k1_ADM_s005
                                 + 0.881909 w_k1_AXP_s005
                                 + 0.659359 w_k1_DIS_s005
                                 + 0.755998 w_k1_FDX_s005
                                 + 0.98854 w_k1_IBM_s005
                                 + 0.677272 w_k1_INTC_s005
                                 + 0.708662 w_k1_KO_s005
                                 + 1.01634 w_k1_MAT_s005
                                 + 0.722935 w_k1_MSFT_s005
                                 + 0.811079 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.99694 w_k1_ADM_s005
                                 + 0.973186 w_k1_AXP_s005
                                 + 0.930356 w_k1_DIS_s005
                                 + 0.924699 w_k1_FDX_s005
                                 + 0.98802 w_k1_IBM_s005
                                 + 0.946585 w_k1_INTC_s005
                                 + 0.86212 w_k1_KO_s005 + 0.79141 w_k1_MAT_s005
                                 + 0.906986 w_k1_MSFT_s005
                                 + 0.871205 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.11365 w_k1_ADM_s005
                                 + 1.20469 w_k1_AXP_s005
                                 + 1.19304 w_k1_DIS_s005 + 1.0023 w_k1_FDX_s005
                                 + 1.74847 w_k1_IBM_s005
                                 + 0.765209 w_k1_INTC_s005
                                 + 1.21181 w_k1_KO_s005 + 1.75911 w_k1_MAT_s005
                                 + 0.917559 w_k1_MSFT_s005
                                 + 1.01751 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.950493 w_k1_ADM_s005
                                 + 1.36619 w_k1_AXP_s005
                                 + 0.895853 w_k1_DIS_s005
                                 + 0.988679 w_k1_FDX_s005
                                 + 1.35507 w_k1_IBM_s005
                                 + 1.15178 w_k1_INTC_s005
                                 + 0.882512 w_k1_KO_s005
                                 + 1.53009 w_k1_MAT_s005
                                 + 1.02012 w_k1_MSFT_s005
                                 + 0.958347 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.11641 w_k1_ADM_s006
                                 + 0.962377 w_k1_AXP_s006 + 0.834 w_k1_DIS_s006
                                 + 0.839035 w_k1_FDX_s006
                                 + 1.2573 w_k1_IBM_s006
                                 + 0.916547 w_k1_INTC_s006
                                 + 1.02427 w_k1_KO_s006 + 1.108 w_k1_MAT_s006
                                 + 0.793794 w_k1_MSFT_s006
                                 + 1.01057 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.47169 w_k1_ADM_s006
                                 + 1.14036 w_k1_AXP_s006
                                 + 1.23669 w_k1_DIS_s006
                                 + 1.34333 w_k1_FDX_s006
                                 + 1.03181 w_k1_IBM_s006
                                 + 1.17626 w_k1_INTC_s006
                                 + 1.26674 w_k1_KO_s006
                                 + 0.916158 w_k1_MAT_s006
                                 + 1.21719 w_k1_MSFT_s006
                                 + 1.12387 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.743539 w_k1_ADM_s006
                                 + 0.65634 w_k1_AXP_s006
                                 + 0.532345 w_k1_DIS_s006
                                 + 0.743758 w_k1_FDX_s006
                                 + 0.602703 w_k1_IBM_s006
                                 + 0.806262 w_k1_INTC_s006
                                 + 0.642484 w_k1_KO_s006
                                 + 0.768789 w_k1_MAT_s006
                                 + 0.719877 w_k1_MSFT_s006
                                 + 0.693138 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.26398 w_k1_ADM_s006
                                 + 0.63301 w_k1_AXP_s006
                                 + 0.767565 w_k1_DIS_s006
                                 + 0.839463 w_k1_FDX_s006
                                 + 0.674479 w_k1_IBM_s006
                                 + 0.869546 w_k1_INTC_s006
                                 + 0.79322 w_k1_KO_s006
                                 + 0.709388 w_k1_MAT_s006
                                 + 0.980254 w_k1_MSFT_s006
                                 + 0.944923 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.883093 w_k1_ADM_s006
                                 + 1.08837 w_k1_AXP_s006
                                 + 0.931543 w_k1_DIS_s006
                                 + 1.05937 w_k1_FDX_s006
                                 + 1.09704 w_k1_IBM_s006
                                 + 1.19876 w_k1_INTC_s006
                                 + 0.884037 w_k1_KO_s006
                                 + 1.22229 w_k1_MAT_s006
                                 + 0.966984 w_k1_MSFT_s006
                                 + 0.988351 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.987958 w_k1_ADM_s006
                                 + 0.940524 w_k1_AXP_s006
                                 + 0.659571 w_k1_DIS_s006
                                 + 0.912817 w_k1_FDX_s006
                                 + 0.727918 w_k1_IBM_s006
                                 + 0.788714 w_k1_INTC_s006
                                 + 0.811663 w_k1_KO_s006
                                 + 1.14598 w_k1_MAT_s006
                                 + 0.824287 w_k1_MSFT_s006
                                 + 0.975476 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.04283 w_k1_ADM_s006
                                 + 1.2726 w_k1_AXP_s006 + 1.47493 w_k1_DIS_s006
                                 + 1.24775 w_k1_FDX_s006
                                 + 1.53565 w_k1_IBM_s006
                                 + 1.36594 w_k1_INTC_s006
                                 + 1.55693 w_k1_KO_s006 + 1.80054 w_k1_MAT_s006
                                 + 1.16956 w_k1_MSFT_s006
                                 + 1.26286 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.991779 w_k1_ADM_s006
                                 + 0.974931 w_k1_AXP_s006
                                 + 0.97322 w_k1_DIS_s006
                                 + 1.02686 w_k1_FDX_s006
                                 + 0.823504 w_k1_IBM_s006
                                 + 1.17651 w_k1_INTC_s006
                                 + 1.02854 w_k1_KO_s006
                                 + 0.826064 w_k1_MAT_s006
                                 + 1.09452 w_k1_MSFT_s006
                                 + 0.975239 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.840779 w_k1_ADM_s007
                                 + 0.878652 w_k1_AXP_s007
                                 + 1.09929 w_k1_DIS_s007
                                 + 0.985703 w_k1_FDX_s007
                                 + 0.812556 w_k1_IBM_s007
                                 + 0.944093 w_k1_INTC_s007
                                 + 1.06368 w_k1_KO_s007
                                 + 0.696444 w_k1_MAT_s007
                                 + 1.04423 w_k1_MSFT_s007
                                 + 1.01662 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.06661 w_k1_ADM_s007
                                 + 1.10563 w_k1_AXP_s007
                                 + 1.55377 w_k1_DIS_s007
                                 + 1.22169 w_k1_FDX_s007
                                 + 0.836657 w_k1_IBM_s007
                                 + 1.02739 w_k1_INTC_s007
                                 + 1.16548 w_k1_KO_s007 + 1.21479 w_k1_MAT_s007
                                 + 1.29057 w_k1_MSFT_s007
                                 + 1.17423 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.735983 w_k1_ADM_s007
                                 + 0.65861 w_k1_AXP_s007
                                 + 0.726732 w_k1_DIS_s007
                                 + 0.856562 w_k1_FDX_s007
                                 + 0.62585 w_k1_IBM_s007
                                 + 0.785972 w_k1_INTC_s007
                                 + 0.817318 w_k1_KO_s007
                                 + 0.765242 w_k1_MAT_s007
                                 + 1.06853 w_k1_MSFT_s007
                                 + 0.945222 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.24101 w_k1_ADM_s007
                                 + 1.36122 w_k1_AXP_s007
                                 + 1.24728 w_k1_DIS_s007
                                 + 1.21096 w_k1_FDX_s007
                                 + 0.959786 w_k1_IBM_s007
                                 + 1.05265 w_k1_INTC_s007
                                 + 1.09731 w_k1_KO_s007
                                 + 0.964102 w_k1_MAT_s007
                                 + 1.15513 w_k1_MSFT_s007
                                 + 1.25471 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.749513 w_k1_ADM_s007
                                 + 0.738449 w_k1_AXP_s007
                                 + 0.727008 w_k1_DIS_s007
                                 + 0.770937 w_k1_FDX_s007
                                 + 1.17181 w_k1_IBM_s007
                                 + 0.845925 w_k1_INTC_s007
                                 + 0.864316 w_k1_KO_s007
                                 + 1.08758 w_k1_MAT_s007
                                 + 0.940431 w_k1_MSFT_s007
                                 + 0.921256 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.28034 w_k1_ADM_s007
                                 + 1.07174 w_k1_AXP_s007 + 1.4812 w_k1_DIS_s007
                                 + 1.13914 w_k1_FDX_s007
                                 + 1.26782 w_k1_IBM_s007
                                 + 1.04338 w_k1_INTC_s007
                                 + 1.59335 w_k1_KO_s007 + 1.37227 w_k1_MAT_s007
                                 + 1.25989 w_k1_MSFT_s007
                                 + 1.20768 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.981421 w_k1_ADM_s007
                                 + 1.11336 w_k1_AXP_s007
                                 + 0.916389 w_k1_DIS_s007
                                 + 0.985464 w_k1_FDX_s007
                                 + 1.18274 w_k1_IBM_s007
                                 + 0.949347 w_k1_INTC_s007
                                 + 1.18177 w_k1_KO_s007 + 1.44116 w_k1_MAT_s007
                                 + 1.15549 w_k1_MSFT_s007 + 1.1939 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.99494 w_k1_ADM_s007
                                 + 0.931805 w_k1_AXP_s007
                                 + 1.22628 w_k1_DIS_s007
                                 + 0.926805 w_k1_FDX_s007
                                 + 1.78219 w_k1_IBM_s007
                                 + 0.773794 w_k1_INTC_s007
                                 + 1.30372 w_k1_KO_s007 + 1.18144 w_k1_MAT_s007
                                 + 1.10581 w_k1_MSFT_s007
                                 + 1.15458 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.23263 w_k1_ADM_s010
                                 + 1.13138 w_k1_AXP_s010
                                 + 1.34669 w_k1_DIS_s010
                                 + 1.20573 w_k1_FDX_s010
                                 + 1.18662 w_k1_IBM_s010
                                 + 1.10261 w_k1_INTC_s010
                                 + 1.26648 w_k1_KO_s010 + 1.30409 w_k1_MAT_s010
                                 + 1.19235 w_k1_MSFT_s010
                                 + 1.14378 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.36492 w_k1_ADM_s010
                                 + 1.35883 w_k1_AXP_s010
                                 + 1.62906 w_k1_DIS_s010
                                 + 1.28877 w_k1_FDX_s010
                                 + 1.48806 w_k1_IBM_s010
                                 + 1.16104 w_k1_INTC_s010
                                 + 1.43695 w_k1_KO_s010 + 1.33715 w_k1_MAT_s010
                                 + 1.22471 w_k1_MSFT_s010
                                 + 1.27541 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.03948 w_k1_ADM_s010
                                 + 1.10776 w_k1_AXP_s010
                                 + 0.954783 w_k1_DIS_s010
                                 + 0.981657 w_k1_FDX_s010
                                 + 1.12234 w_k1_IBM_s010
                                 + 0.996636 w_k1_INTC_s010
                                 + 0.957974 w_k1_KO_s010
                                 + 1.08429 w_k1_MAT_s010
                                 + 0.894954 w_k1_MSFT_s010
                                 + 1.01319 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.48167 w_k1_ADM_s010
                                 + 1.29614 w_k1_AXP_s010
                                 + 1.48751 w_k1_DIS_s010
                                 + 1.43875 w_k1_FDX_s010
                                 + 0.969742 w_k1_IBM_s010
                                 + 1.09646 w_k1_INTC_s010
                                 + 1.06901 w_k1_KO_s010 + 1.05845 w_k1_MAT_s010
                                 + 1.21643 w_k1_MSFT_s010
                                 + 1.34777 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.872044 w_k1_ADM_s010
                                 + 0.749691 w_k1_AXP_s010
                                 + 0.66588 w_k1_DIS_s010
                                 + 0.791547 w_k1_FDX_s010
                                 + 0.91996 w_k1_IBM_s010
                                 + 0.785162 w_k1_INTC_s010
                                 + 0.779413 w_k1_KO_s010
                                 + 0.955575 w_k1_MAT_s010
                                 + 0.898812 w_k1_MSFT_s010
                                 + 0.868882 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.711619 w_k1_ADM_s010
                                 + 0.976454 w_k1_AXP_s010
                                 + 1.00465 w_k1_DIS_s010
                                 + 0.926551 w_k1_FDX_s010
                                 + 1.56674 w_k1_IBM_s010
                                 + 0.967561 w_k1_INTC_s010
                                 + 1.23318 w_k1_KO_s010 + 1.43465 w_k1_MAT_s010
                                 + 0.963576 w_k1_MSFT_s010
                                 + 1.03651 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.01578 w_k1_ADM_s010
                                 + 0.871899 w_k1_AXP_s010
                                 + 0.773249 w_k1_DIS_s010
                                 + 0.954209 w_k1_FDX_s010
                                 + 0.714206 w_k1_IBM_s010
                                 + 0.947909 w_k1_INTC_s010
                                 + 0.732984 w_k1_KO_s010
                                 + 0.57733 w_k1_MAT_s010
                                 + 0.85394 w_k1_MSFT_s010
                                 + 0.916179 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.825726 w_k1_ADM_s010
                                 + 0.909575 w_k1_AXP_s010
                                 + 0.997595 w_k1_DIS_s010
                                 + 1.04072 w_k1_FDX_s010
                                 + 0.794278 w_k1_IBM_s010
                                 + 1.0997 w_k1_INTC_s010
                                 + 0.967839 w_k1_KO_s010
                                 + 1.00767 w_k1_MAT_s010
                                 + 1.04487 w_k1_MSFT_s010
                                 + 1.03306 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.743451 w_k1_ADM_s011
                                 + 0.538467 w_k1_AXP_s011
                                 + 0.592918 w_k1_DIS_s011
                                 + 0.638036 w_k1_FDX_s011
                                 + 0.940876 w_k1_IBM_s011
                                 + 0.735374 w_k1_INTC_s011
                                 + 0.880464 w_k1_KO_s011
                                 + 0.859739 w_k1_MAT_s011
                                 + 0.675665 w_k1_MSFT_s011
                                 + 0.834279 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.09304 w_k1_ADM_s011
                                 + 1.48295 w_k1_AXP_s011
                                 + 1.75428 w_k1_DIS_s011
                                 + 1.25412 w_k1_FDX_s011
                                 + 1.52631 w_k1_IBM_s011
                                 + 1.17276 w_k1_INTC_s011
                                 + 1.47741 w_k1_KO_s011 + 1.99052 w_k1_MAT_s011
                                 + 1.02852 w_k1_MSFT_s011
                                 + 1.39418 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.73101 w_k1_ADM_s011
                                 + 1.35539 w_k1_AXP_s011
                                 + 1.74695 w_k1_DIS_s011
                                 + 1.54435 w_k1_FDX_s011
                                 + 0.862501 w_k1_IBM_s011
                                 + 1.22224 w_k1_INTC_s011
                                 + 1.44013 w_k1_KO_s011 + 0.94037 w_k1_MAT_s011
                                 + 1.42144 w_k1_MSFT_s011
                                 + 1.33381 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.20169 w_k1_ADM_s011
                                 + 0.92213 w_k1_AXP_s011
                                 + 0.973525 w_k1_DIS_s011
                                 + 1.02811 w_k1_FDX_s011
                                 + 0.677282 w_k1_IBM_s011
                                 + 0.986519 w_k1_INTC_s011
                                 + 0.849196 w_k1_KO_s011
                                 + 0.791081 w_k1_MAT_s011
                                 + 0.842493 w_k1_MSFT_s011
                                 + 0.901525 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.951423 w_k1_ADM_s011
                                 + 0.91497 w_k1_AXP_s011
                                 + 0.981404 w_k1_DIS_s011
                                 + 0.821311 w_k1_FDX_s011
                                 + 1.46071 w_k1_IBM_s011
                                 + 0.743188 w_k1_INTC_s011
                                 + 1.19576 w_k1_KO_s011 + 1.32609 w_k1_MAT_s011
                                 + 0.788938 w_k1_MSFT_s011
                                 + 0.950074 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.856096 w_k1_ADM_s011
                                 + 0.913286 w_k1_AXP_s011
                                 + 0.77292 w_k1_DIS_s011
                                 + 0.81402 w_k1_FDX_s011 + 1.0046 w_k1_IBM_s011
                                 + 0.776406 w_k1_INTC_s011
                                 + 0.984242 w_k1_KO_s011 + 1.0151 w_k1_MAT_s011
                                 + 0.773288 w_k1_MSFT_s011
                                 + 0.885393 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.07611 w_k1_ADM_s011
                                 + 1.24731 w_k1_AXP_s011
                                 + 1.97922 w_k1_DIS_s011
                                 + 1.36615 w_k1_FDX_s011
                                 + 0.995512 w_k1_IBM_s011
                                 + 1.36286 w_k1_INTC_s011
                                 + 1.22389 w_k1_KO_s011
                                 + 0.859207 w_k1_MAT_s011
                                 + 1.33061 w_k1_MSFT_s011
                                 + 1.17301 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.01221 w_k1_ADM_s011
                                 + 0.973023 w_k1_AXP_s011
                                 + 1.21092 w_k1_DIS_s011
                                 + 1.06854 w_k1_FDX_s011
                                 + 1.02582 w_k1_IBM_s011
                                 + 0.994232 w_k1_INTC_s011
                                 + 1.1342 w_k1_KO_s011 + 1.1805 w_k1_MAT_s011
                                 + 1.1983 w_k1_MSFT_s011 + 1.05559 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.93653 w_k1_ADM_s012
                                 + 0.905945 w_k1_AXP_s012
                                 + 1.02153 w_k1_DIS_s012
                                 + 1.01367 w_k1_FDX_s012
                                 + 0.931504 w_k1_IBM_s012
                                 + 1.11369 w_k1_INTC_s012
                                 + 1.03208 w_k1_KO_s012 + 1.18279 w_k1_MAT_s012
                                 + 1.07568 w_k1_MSFT_s012
                                 + 0.980751 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.825108 w_k1_ADM_s012
                                 + 0.719454 w_k1_AXP_s012
                                 + 0.72486 w_k1_DIS_s012
                                 + 0.766246 w_k1_FDX_s012
                                 + 0.90886 w_k1_IBM_s012
                                 + 0.863188 w_k1_INTC_s012
                                 + 0.783559 w_k1_KO_s012
                                 + 0.695291 w_k1_MAT_s012
                                 + 0.837705 w_k1_MSFT_s012
                                 + 0.94354 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.02602 w_k1_ADM_s012
                                 + 1.05727 w_k1_AXP_s012
                                 + 0.710041 w_k1_DIS_s012
                                 + 0.889715 w_k1_FDX_s012
                                 + 0.783274 w_k1_IBM_s012
                                 + 0.910118 w_k1_INTC_s012
                                 + 0.727385 w_k1_KO_s012
                                 + 0.733647 w_k1_MAT_s012
                                 + 0.711889 w_k1_MSFT_s012
                                 + 0.912413 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.0694 w_k1_ADM_s012
                                 + 0.766624 w_k1_AXP_s012
                                 + 0.88514 w_k1_DIS_s012
                                 + 0.998297 w_k1_FDX_s012
                                 + 0.755281 w_k1_IBM_s012
                                 + 1.01114 w_k1_INTC_s012
                                 + 0.802217 w_k1_KO_s012 + 0.6218 w_k1_MAT_s012
                                 + 0.857735 w_k1_MSFT_s012
                                 + 0.90163 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.09354 w_k1_ADM_s012
                                 + 1.07391 w_k1_AXP_s012
                                 + 0.812096 w_k1_DIS_s012
                                 + 0.868713 w_k1_FDX_s012
                                 + 1.19667 w_k1_IBM_s012
                                 + 0.988881 w_k1_INTC_s012
                                 + 0.965416 w_k1_KO_s012 + 1.0636 w_k1_MAT_s012
                                 + 0.776687 w_k1_MSFT_s012
                                 + 0.977663 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.12192 w_k1_ADM_s012
                                 + 1.41792 w_k1_AXP_s012
                                 + 1.37328 w_k1_DIS_s012
                                 + 1.26303 w_k1_FDX_s012
                                 + 1.23155 w_k1_IBM_s012
                                 + 1.23105 w_k1_INTC_s012
                                 + 1.26294 w_k1_KO_s012 + 1.20103 w_k1_MAT_s012
                                 + 1.19974 w_k1_MSFT_s012
                                 + 1.26785 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.929592 w_k1_ADM_s012
                                 + 0.762838 w_k1_AXP_s012
                                 + 0.486845 w_k1_DIS_s012
                                 + 0.730334 w_k1_FDX_s012
                                 + 0.933895 w_k1_IBM_s012
                                 + 0.840653 w_k1_INTC_s012
                                 + 0.82231 w_k1_KO_s012
                                 + 0.831987 w_k1_MAT_s012
                                 + 0.691684 w_k1_MSFT_s012
                                 + 0.81411 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.808949 w_k1_ADM_s012
                                 + 0.747267 w_k1_AXP_s012
                                 + 0.759864 w_k1_DIS_s012
                                 + 0.892749 w_k1_FDX_s012
                                 + 0.609736 w_k1_IBM_s012
                                 + 0.909414 w_k1_INTC_s012
                                 + 0.761239 w_k1_KO_s012
                                 + 0.569795 w_k1_MAT_s012
                                 + 0.820227 w_k1_MSFT_s012
                                 + 0.891831 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.901913 w_k1_ADM_s013
                                 + 1.19342 w_k1_AXP_s013
                                 + 1.06201 w_k1_DIS_s013
                                 + 1.05365 w_k1_FDX_s013
                                 + 1.06926 w_k1_IBM_s013
                                 + 1.05356 w_k1_INTC_s013
                                 + 1.00976 w_k1_KO_s013
                                 + 0.868943 w_k1_MAT_s013
                                 + 1.09579 w_k1_MSFT_s013
                                 + 0.936876 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.677437 w_k1_ADM_s013
                                 + 0.758586 w_k1_AXP_s013
                                 + 0.700794 w_k1_DIS_s013
                                 + 0.841041 w_k1_FDX_s013
                                 + 0.651754 w_k1_IBM_s013
                                 + 0.932357 w_k1_INTC_s013
                                 + 0.708336 w_k1_KO_s013
                                 + 0.399668 w_k1_MAT_s013
                                 + 0.873397 w_k1_MSFT_s013
                                 + 0.853489 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.910393 w_k1_ADM_s013
                                 + 1.41414 w_k1_AXP_s013
                                 + 1.60933 w_k1_DIS_s013
                                 + 1.16496 w_k1_FDX_s013
                                 + 1.44694 w_k1_IBM_s013
                                 + 1.00837 w_k1_INTC_s013
                                 + 1.25125 w_k1_KO_s013 + 1.58295 w_k1_MAT_s013
                                 + 1.31703 w_k1_MSFT_s013
                                 + 1.22393 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.05102 w_k1_ADM_s013
                                 + 1.42564 w_k1_AXP_s013
                                 + 1.62891 w_k1_DIS_s013
                                 + 1.33923 w_k1_FDX_s013
                                 + 0.942987 w_k1_IBM_s013
                                 + 1.02881 w_k1_INTC_s013
                                 + 1.04136 w_k1_KO_s013 + 1.29036 w_k1_MAT_s013
                                 + 1.23179 w_k1_MSFT_s013
                                 + 1.13419 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.761316 w_k1_ADM_s013
                                 + 0.825968 w_k1_AXP_s013
                                 + 0.821957 w_k1_DIS_s013
                                 + 0.939635 w_k1_FDX_s013
                                 + 1.13285 w_k1_IBM_s013
                                 + 0.733521 w_k1_INTC_s013
                                 + 0.987938 w_k1_KO_s013
                                 + 1.00988 w_k1_MAT_s013
                                 + 0.860429 w_k1_MSFT_s013
                                 + 0.954354 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.0819 w_k1_ADM_s013
                                 + 1.2702 w_k1_AXP_s013 + 1.11676 w_k1_DIS_s013
                                 + 1.1204 w_k1_FDX_s013
                                 + 0.796188 w_k1_IBM_s013
                                 + 0.943954 w_k1_INTC_s013
                                 + 0.922836 w_k1_KO_s013
                                 + 0.64619 w_k1_MAT_s013
                                 + 1.05706 w_k1_MSFT_s013
                                 + 0.874787 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.968327 w_k1_ADM_s013
                                 + 0.826627 w_k1_AXP_s013
                                 + 1.26036 w_k1_DIS_s013
                                 + 1.11549 w_k1_FDX_s013
                                 + 0.807768 w_k1_IBM_s013
                                 + 1.13682 w_k1_INTC_s013
                                 + 0.975107 w_k1_KO_s013
                                 + 0.498242 w_k1_MAT_s013
                                 + 1.2808 w_k1_MSFT_s013 + 0.95091 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.936777 w_k1_ADM_s013
                                 + 1.00518 w_k1_AXP_s013
                                 + 0.896391 w_k1_DIS_s013
                                 + 0.839417 w_k1_FDX_s013
                                 + 1.06553 w_k1_IBM_s013
                                 + 0.767391 w_k1_INTC_s013
                                 + 0.958828 w_k1_KO_s013
                                 + 0.945666 w_k1_MAT_s013
                                 + 0.873373 w_k1_MSFT_s013
                                 + 0.939011 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.12543 w_k1_ADM_s014
                                 + 1.14742 w_k1_AXP_s014
                                 + 0.956674 w_k1_DIS_s014
                                 + 1.01888 w_k1_FDX_s014
                                 + 0.966959 w_k1_IBM_s014
                                 + 1.00024 w_k1_INTC_s014
                                 + 0.968077 w_k1_KO_s014
                                 + 0.961705 w_k1_MAT_s014
                                 + 1.06582 w_k1_MSFT_s014
                                 + 1.07214 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.848458 w_k1_ADM_s014
                                 + 0.813265 w_k1_AXP_s014
                                 + 0.652105 w_k1_DIS_s014
                                 + 0.777815 w_k1_FDX_s014
                                 + 0.774485 w_k1_IBM_s014
                                 + 0.850569 w_k1_INTC_s014
                                 + 0.774357 w_k1_KO_s014
                                 + 0.558899 w_k1_MAT_s014
                                 + 0.872359 w_k1_MSFT_s014
                                 + 0.883647 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.20377 w_k1_ADM_s014
                                 + 1.61677 w_k1_AXP_s014
                                 + 1.60609 w_k1_DIS_s014
                                 + 1.40403 w_k1_FDX_s014
                                 + 1.12297 w_k1_IBM_s014
                                 + 1.38948 w_k1_INTC_s014
                                 + 1.65429 w_k1_KO_s014 + 1.58462 w_k1_MAT_s014
                                 + 1.58822 w_k1_MSFT_s014
                                 + 1.31111 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.995762 w_k1_ADM_s014
                                 + 0.756009 w_k1_AXP_s014
                                 + 0.985592 w_k1_DIS_s014
                                 + 1.02343 w_k1_FDX_s014
                                 + 0.779031 w_k1_IBM_s014
                                 + 1.10306 w_k1_INTC_s014
                                 + 1.11098 w_k1_KO_s014
                                 + 0.601997 w_k1_MAT_s014
                                 + 1.22775 w_k1_MSFT_s014
                                 + 1.01424 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.917908 w_k1_ADM_s014
                                 + 0.976263 w_k1_AXP_s014
                                 + 1.00042 w_k1_DIS_s014
                                 + 0.961347 w_k1_FDX_s014
                                 + 1.35533 w_k1_IBM_s014
                                 + 0.945017 w_k1_INTC_s014
                                 + 1.30521 w_k1_KO_s014 + 1.18927 w_k1_MAT_s014
                                 + 1.14379 w_k1_MSFT_s014 + 1.0246 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.01535 w_k1_ADM_s014
                                 + 0.950481 w_k1_AXP_s014
                                 + 0.792528 w_k1_DIS_s014
                                 + 0.865654 w_k1_FDX_s014
                                 + 1.08921 w_k1_IBM_s014
                                 + 0.950411 w_k1_INTC_s014
                                 + 1.00902 w_k1_KO_s014 + 1.05661 w_k1_MAT_s014
                                 + 1.1253 w_k1_MSFT_s014 + 1.02342 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.06939 w_k1_ADM_s014
                                 + 1.07059 w_k1_AXP_s014
                                 + 1.08731 w_k1_DIS_s014
                                 + 1.18923 w_k1_FDX_s014
                                 + 0.827155 w_k1_IBM_s014
                                 + 1.29318 w_k1_INTC_s014
                                 + 0.962952 w_k1_KO_s014
                                 + 0.873599 w_k1_MAT_s014
                                 + 1.17329 w_k1_MSFT_s014
                                 + 1.12051 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.976915 w_k1_ADM_s014
                                 + 1.15391 w_k1_AXP_s014
                                 + 1.28028 w_k1_DIS_s014
                                 + 1.16242 w_k1_FDX_s014
                                 + 1.37271 w_k1_IBM_s014
                                 + 1.31893 w_k1_INTC_s014
                                 + 1.39969 w_k1_KO_s014 + 1.33905 w_k1_MAT_s014
                                 + 1.30999 w_k1_MSFT_s014
                                 + 1.12665 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.939035 w_k1_ADM_s015
                                 + 0.791842 w_k1_AXP_s015
                                 + 0.973916 w_k1_DIS_s015
                                 + 1.01183 w_k1_FDX_s015
                                 + 0.77624 w_k1_IBM_s015
                                 + 1.00444 w_k1_INTC_s015
                                 + 0.926634 w_k1_KO_s015
                                 + 0.744894 w_k1_MAT_s015
                                 + 1.09719 w_k1_MSFT_s015
                                 + 0.990963 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.01865 w_k1_ADM_s015
                                 + 0.985482 w_k1_AXP_s015
                                 + 1.05681 w_k1_DIS_s015
                                 + 1.01983 w_k1_FDX_s015
                                 + 1.16561 w_k1_IBM_s015
                                 + 1.13149 w_k1_INTC_s015
                                 + 1.06031 w_k1_KO_s015 + 1.03361 w_k1_MAT_s015
                                 + 1.05685 w_k1_MSFT_s015
                                 + 0.967249 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.00951 w_k1_ADM_s015
                                 + 0.8946 w_k1_AXP_s015
                                 + 0.786473 w_k1_DIS_s015
                                 + 0.923203 w_k1_FDX_s015
                                 + 0.95904 w_k1_IBM_s015
                                 + 0.855931 w_k1_INTC_s015
                                 + 1.00429 w_k1_KO_s015 + 0.75463 w_k1_MAT_s015
                                 + 0.892368 w_k1_MSFT_s015
                                 + 0.954433 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.03993 w_k1_ADM_s015
                                 + 0.942885 w_k1_AXP_s015
                                 + 0.773177 w_k1_DIS_s015
                                 + 0.796372 w_k1_FDX_s015
                                 + 1.55937 w_k1_IBM_s015
                                 + 0.884664 w_k1_INTC_s015
                                 + 1.1464 w_k1_KO_s015 + 1.27708 w_k1_MAT_s015
                                 + 0.869488 w_k1_MSFT_s015
                                 + 0.948174 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.40745 w_k1_ADM_s015
                                 + 1.37945 w_k1_AXP_s015
                                 + 1.78845 w_k1_DIS_s015
                                 + 1.41421 w_k1_FDX_s015
                                 + 1.06248 w_k1_IBM_s015
                                 + 1.25734 w_k1_INTC_s015
                                 + 1.23815 w_k1_KO_s015 + 1.42275 w_k1_MAT_s015
                                 + 1.40392 w_k1_MSFT_s015
                                 + 1.24525 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.986625 w_k1_ADM_s015
                                 + 0.866953 w_k1_AXP_s015
                                 + 1.34196 w_k1_DIS_s015
                                 + 1.12242 w_k1_FDX_s015
                                 + 0.938573 w_k1_IBM_s015
                                 + 1.13629 w_k1_INTC_s015
                                 + 1.02091 w_k1_KO_s015
                                 + 0.773516 w_k1_MAT_s015
                                 + 1.17479 w_k1_MSFT_s015
                                 + 1.06356 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.833425 w_k1_ADM_s015
                                 + 0.683868 w_k1_AXP_s015
                                 + 0.616677 w_k1_DIS_s015
                                 + 0.674117 w_k1_FDX_s015
                                 + 0.8581 w_k1_IBM_s015
                                 + 0.740547 w_k1_INTC_s015
                                 + 0.757535 w_k1_KO_s015
                                 + 0.781433 w_k1_MAT_s015
                                 + 0.793299 w_k1_MSFT_s015
                                 + 0.872164 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.07419 w_k1_ADM_s015
                                 + 1.51843 w_k1_AXP_s015 + 1.3115 w_k1_DIS_s015
                                 + 1.23812 w_k1_FDX_s015
                                 + 1.46793 w_k1_IBM_s015
                                 + 1.23812 w_k1_INTC_s015
                                 + 1.22483 w_k1_KO_s015 + 1.8853 w_k1_MAT_s015
                                 + 1.0941 w_k1_MSFT_s015 + 1.26045 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.38151 w_k1_ADM_s016
                                 + 0.984432 w_k1_AXP_s016
                                 + 0.786465 w_k1_DIS_s016
                                 + 0.942564 w_k1_FDX_s016
                                 + 0.80777 w_k1_IBM_s016
                                 + 0.859436 w_k1_INTC_s016
                                 + 0.860092 w_k1_KO_s016
                                 + 1.00661 w_k1_MAT_s016
                                 + 0.814167 w_k1_MSFT_s016
                                 + 0.973676 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.12613 w_k1_ADM_s016
                                 + 0.868921 w_k1_AXP_s016
                                 + 0.905413 w_k1_DIS_s016
                                 + 0.990816 w_k1_FDX_s016
                                 + 0.641255 w_k1_IBM_s016
                                 + 0.961138 w_k1_INTC_s016
                                 + 0.727111 w_k1_KO_s016
                                 + 0.813828 w_k1_MAT_s016
                                 + 0.854486 w_k1_MSFT_s016
                                 + 0.908283 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.993043 w_k1_ADM_s016
                                 + 0.935567 w_k1_AXP_s016
                                 + 0.624907 w_k1_DIS_s016
                                 + 0.741005 w_k1_FDX_s016
                                 + 0.913425 w_k1_IBM_s016
                                 + 0.6681 w_k1_INTC_s016
                                 + 0.713439 w_k1_KO_s016
                                 + 0.970141 w_k1_MAT_s016
                                 + 0.664475 w_k1_MSFT_s016
                                 + 0.895441 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.34557 w_k1_ADM_s016
                                 + 1.56254 w_k1_AXP_s016
                                 + 1.66824 w_k1_DIS_s016
                                 + 1.40253 w_k1_FDX_s016
                                 + 0.928563 w_k1_IBM_s016
                                 + 1.11885 w_k1_INTC_s016
                                 + 1.04193 w_k1_KO_s016
                                 + 0.812276 w_k1_MAT_s016
                                 + 1.22378 w_k1_MSFT_s016
                                 + 1.19559 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.973074 w_k1_ADM_s016
                                 + 0.952688 w_k1_AXP_s016
                                 + 1.12262 w_k1_DIS_s016
                                 + 0.984748 w_k1_FDX_s016
                                 + 0.82848 w_k1_IBM_s016
                                 + 1.02465 w_k1_INTC_s016
                                 + 0.983753 w_k1_KO_s016
                                 + 0.714251 w_k1_MAT_s016
                                 + 1.01258 w_k1_MSFT_s016
                                 + 1.06874 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.38779 w_k1_ADM_s016
                                 + 1.15639 w_k1_AXP_s016
                                 + 1.14254 w_k1_DIS_s016
                                 + 1.09858 w_k1_FDX_s016
                                 + 0.904515 w_k1_IBM_s016
                                 + 1.07221 w_k1_INTC_s016
                                 + 0.877644 w_k1_KO_s016
                                 + 0.929414 w_k1_MAT_s016
                                 + 0.969072 w_k1_MSFT_s016
                                 + 1.04528 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.41747 w_k1_ADM_s016
                                 + 1.56803 w_k1_AXP_s016
                                 + 1.57033 w_k1_DIS_s016
                                 + 1.33927 w_k1_FDX_s016
                                 + 1.74834 w_k1_IBM_s016
                                 + 1.19822 w_k1_INTC_s016
                                 + 1.33931 w_k1_KO_s016 + 1.50628 w_k1_MAT_s016
                                 + 1.15269 w_k1_MSFT_s016
                                 + 1.16764 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.978121 w_k1_ADM_s016
                                 + 1.17778 w_k1_AXP_s016 + 1.18 w_k1_DIS_s016
                                 + 0.971803 w_k1_FDX_s016
                                 + 1.3843 w_k1_IBM_s016
                                 + 1.10393 w_k1_INTC_s016
                                 + 1.08191 w_k1_KO_s016 + 1.05057 w_k1_MAT_s016
                                 + 0.854731 w_k1_MSFT_s016
                                 + 1.08895 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.2364 w_k1_ADM_s017
                                 + 1.25283 w_k1_AXP_s017
                                 + 1.08861 w_k1_DIS_s017
                                 + 0.989098 w_k1_FDX_s017
                                 + 1.56096 w_k1_IBM_s017
                                 + 1.10504 w_k1_INTC_s017
                                 + 1.28124 w_k1_KO_s017 + 1.2218 w_k1_MAT_s017
                                 + 0.919623 w_k1_MSFT_s017
                                 + 1.19226 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.25891 w_k1_ADM_s017
                                 + 1.2218 w_k1_AXP_s017 + 1.64669 w_k1_DIS_s017
                                 + 1.38032 w_k1_FDX_s017
                                 + 1.03448 w_k1_IBM_s017
                                 + 1.19454 w_k1_INTC_s017
                                 + 1.09818 w_k1_KO_s017 + 1.00379 w_k1_MAT_s017
                                 + 1.26091 w_k1_MSFT_s017
                                 + 1.18097 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.862961 w_k1_ADM_s017
                                 + 1.00858 w_k1_AXP_s017
                                 + 0.990152 w_k1_DIS_s017
                                 + 1.02125 w_k1_FDX_s017
                                 + 1.00412 w_k1_IBM_s017
                                 + 1.03531 w_k1_INTC_s017
                                 + 1.00607 w_k1_KO_s017
                                 + 0.928625 w_k1_MAT_s017
                                 + 1.01311 w_k1_MSFT_s017
                                 + 1.01558 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.48084 w_k1_ADM_s017
                                 + 1.44313 w_k1_AXP_s017
                                 + 1.62222 w_k1_DIS_s017
                                 + 1.18032 w_k1_FDX_s017 + 1.6354 w_k1_IBM_s017
                                 + 1.28668 w_k1_INTC_s017
                                 + 1.46116 w_k1_KO_s017 + 1.67091 w_k1_MAT_s017
                                 + 1.24784 w_k1_MSFT_s017
                                 + 1.29317 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.18058 w_k1_ADM_s017
                                 + 1.19913 w_k1_AXP_s017
                                 + 1.08554 w_k1_DIS_s017
                                 + 1.14255 w_k1_FDX_s017
                                 + 0.861876 w_k1_IBM_s017
                                 + 0.970544 w_k1_INTC_s017
                                 + 0.901767 w_k1_KO_s017
                                 + 0.862536 w_k1_MAT_s017
                                 + 0.895201 w_k1_MSFT_s017
                                 + 1.06956 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.00873 w_k1_ADM_s017
                                 + 1.05342 w_k1_AXP_s017
                                 + 0.891068 w_k1_DIS_s017
                                 + 0.885555 w_k1_FDX_s017
                                 + 1.04104 w_k1_IBM_s017
                                 + 0.903103 w_k1_INTC_s017
                                 + 1.01775 w_k1_KO_s017 + 1.29589 w_k1_MAT_s017
                                 + 0.886151 w_k1_MSFT_s017
                                 + 1.00037 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.73686 w_k1_ADM_s017
                                 + 0.75835 w_k1_AXP_s017
                                 + 0.69748 w_k1_DIS_s017
                                 + 0.74151 w_k1_FDX_s017
                                 + 0.84823 w_k1_IBM_s017
                                 + 0.808207 w_k1_INTC_s017
                                 + 0.706702 w_k1_KO_s017
                                 + 1.04524 w_k1_MAT_s017
                                 + 0.787015 w_k1_MSFT_s017
                                 + 0.972738 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.943101 w_k1_ADM_s017
                                 + 0.764144 w_k1_AXP_s017
                                 + 0.957888 w_k1_DIS_s017
                                 + 1.01134 w_k1_FDX_s017
                                 + 0.778751 w_k1_IBM_s017
                                 + 1.1926 w_k1_INTC_s017
                                 + 0.951482 w_k1_KO_s017
                                 + 0.812838 w_k1_MAT_s017
                                 + 1.07086 w_k1_MSFT_s017
                                 + 0.957391 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.733776 w_k1_ADM_s100
                                 + 0.88302 w_k1_AXP_s100
                                 + 0.809116 w_k1_DIS_s100
                                 + 0.943984 w_k1_FDX_s100
                                 + 0.726526 w_k1_IBM_s100
                                 + 0.894296 w_k1_INTC_s100
                                 + 0.750957 w_k1_KO_s100
                                 + 0.667198 w_k1_MAT_s100
                                 + 0.789057 w_k1_MSFT_s100
                                 + 0.840261 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.17693 w_k1_ADM_s100
                                 + 0.893679 w_k1_AXP_s100
                                 + 0.999818 w_k1_DIS_s100
                                 + 0.918702 w_k1_FDX_s100
                                 + 1.19258 w_k1_IBM_s100
                                 + 0.957636 w_k1_INTC_s100
                                 + 1.14458 w_k1_KO_s100 + 1.45492 w_k1_MAT_s100
                                 + 0.919587 w_k1_MSFT_s100
                                 + 0.965433 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.941486 w_k1_ADM_s100
                                 + 1.05864 w_k1_AXP_s100
                                 + 1.33232 w_k1_DIS_s100
                                 + 1.07667 w_k1_FDX_s100 + 1.7959 w_k1_IBM_s100
                                 + 0.995932 w_k1_INTC_s100
                                 + 1.34794 w_k1_KO_s100 + 1.89244 w_k1_MAT_s100
                                 + 1.06193 w_k1_MSFT_s100 + 1.0146 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.28151 w_k1_ADM_s100
                                 + 1.1954 w_k1_AXP_s100
                                 + 0.824807 w_k1_DIS_s100
                                 + 0.928583 w_k1_FDX_s100
                                 + 1.20905 w_k1_IBM_s100
                                 + 0.869773 w_k1_INTC_s100
                                 + 0.884509 w_k1_KO_s100
                                 + 1.25526 w_k1_MAT_s100
                                 + 0.866317 w_k1_MSFT_s100
                                 + 0.882095 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.707141 w_k1_ADM_s100
                                 + 0.837603 w_k1_AXP_s100
                                 + 0.693001 w_k1_DIS_s100
                                 + 0.751556 w_k1_FDX_s100
                                 + 0.889247 w_k1_IBM_s100
                                 + 0.835425 w_k1_INTC_s100
                                 + 0.760988 w_k1_KO_s100
                                 + 0.780803 w_k1_MAT_s100
                                 + 0.800837 w_k1_MSFT_s100
                                 + 0.77732 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.3348 w_k1_ADM_s100
                                 + 1.12066 w_k1_AXP_s100
                                 + 1.19992 w_k1_DIS_s100
                                 + 1.09444 w_k1_FDX_s100
                                 + 1.22752 w_k1_IBM_s100
                                 + 1.28224 w_k1_INTC_s100
                                 + 1.14758 w_k1_KO_s100
                                 + 0.951718 w_k1_MAT_s100
                                 + 1.14785 w_k1_MSFT_s100
                                 + 1.09431 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.50662 w_k1_ADM_s100
                                 + 1.7421 w_k1_AXP_s100 + 1.46419 w_k1_DIS_s100
                                 + 1.40821 w_k1_FDX_s100
                                 + 1.23976 w_k1_IBM_s100
                                 + 1.12316 w_k1_INTC_s100
                                 + 1.23725 w_k1_KO_s100 + 1.23946 w_k1_MAT_s100
                                 + 1.03419 w_k1_MSFT_s100
                                 + 1.19465 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.936167 w_k1_ADM_s100
                                 + 1.0551 w_k1_AXP_s100 + 1.17737 w_k1_DIS_s100
                                 + 1.18066 w_k1_FDX_s100
                                 + 0.784061 w_k1_IBM_s100
                                 + 0.89508 w_k1_INTC_s100
                                 + 0.975104 w_k1_KO_s100
                                 + 0.949614 w_k1_MAT_s100
                                 + 1.00788 w_k1_MSFT_s100
                                 + 1.13436 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.55999 w_k1_ADM_s101
                                 + 1.63496 w_k1_AXP_s101
                                 + 1.53816 w_k1_DIS_s101
                                 + 1.38543 w_k1_FDX_s101
                                 + 1.37728 w_k1_IBM_s101
                                 + 1.10573 w_k1_INTC_s101
                                 + 1.41289 w_k1_KO_s101 + 2.40978 w_k1_MAT_s101
                                 + 1.50624 w_k1_MSFT_s101
                                 + 1.30114 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.890096 w_k1_ADM_s101
                                 + 1.03559 w_k1_AXP_s101
                                 + 0.855744 w_k1_DIS_s101
                                 + 0.975846 w_k1_FDX_s101
                                 + 0.884577 w_k1_IBM_s101
                                 + 0.891627 w_k1_INTC_s101
                                 + 0.940643 w_k1_KO_s101
                                 + 1.31463 w_k1_MAT_s101
                                 + 1.16133 w_k1_MSFT_s101
                                 + 0.925652 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.19423 w_k1_ADM_s101
                                 + 1.2547 w_k1_AXP_s101 + 1.59622 w_k1_DIS_s101
                                 + 1.26996 w_k1_FDX_s101
                                 + 0.948121 w_k1_IBM_s101
                                 + 1.15725 w_k1_INTC_s101
                                 + 1.37713 w_k1_KO_s101 + 1.29243 w_k1_MAT_s101
                                 + 1.53712 w_k1_MSFT_s101
                                 + 1.15097 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.05124 w_k1_ADM_s101
                                 + 0.956799 w_k1_AXP_s101
                                 + 1.00599 w_k1_DIS_s101
                                 + 0.990008 w_k1_FDX_s101
                                 + 1.37865 w_k1_IBM_s101
                                 + 1.00673 w_k1_INTC_s101
                                 + 1.14062 w_k1_KO_s101 + 1.87295 w_k1_MAT_s101
                                 + 1.30505 w_k1_MSFT_s101
                                 + 1.12452 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.18796 w_k1_ADM_s101
                                 + 1.09555 w_k1_AXP_s101
                                 + 1.05911 w_k1_DIS_s101
                                 + 0.975316 w_k1_FDX_s101
                                 + 1.83156 w_k1_IBM_s101
                                 + 0.852836 w_k1_INTC_s101
                                 + 1.21078 w_k1_KO_s101 + 2.1177 w_k1_MAT_s101
                                 + 1.07932 w_k1_MSFT_s101
                                 + 1.05443 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.34262 w_k1_ADM_s101
                                 + 1.44472 w_k1_AXP_s101
                                 + 1.04791 w_k1_DIS_s101
                                 + 1.22852 w_k1_FDX_s101
                                 + 0.928507 w_k1_IBM_s101
                                 + 0.942859 w_k1_INTC_s101
                                 + 1.03109 w_k1_KO_s101 + 1.33043 w_k1_MAT_s101
                                 + 1.22042 w_k1_MSFT_s101
                                 + 1.17379 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.40012 w_k1_ADM_s101
                                 + 0.869379 w_k1_AXP_s101
                                 + 1.10177 w_k1_DIS_s101
                                 + 1.02413 w_k1_FDX_s101
                                 + 0.842948 w_k1_IBM_s101
                                 + 1.03965 w_k1_INTC_s101
                                 + 1.08886 w_k1_KO_s101
                                 + 0.975883 w_k1_MAT_s101
                                 + 1.32026 w_k1_MSFT_s101
                                 + 1.01645 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.994375 w_k1_ADM_s101
                                 + 0.757503 w_k1_AXP_s101
                                 + 0.609607 w_k1_DIS_s101
                                 + 0.807824 w_k1_FDX_s101
                                 + 0.802349 w_k1_IBM_s101
                                 + 0.616509 w_k1_INTC_s101
                                 + 0.765348 w_k1_KO_s101
                                 + 0.887287 w_k1_MAT_s101
                                 + 0.868852 w_k1_MSFT_s101
                                 + 0.876787 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.789414 w_k1_ADM_s102
                                 + 0.576225 w_k1_AXP_s102
                                 + 0.408829 w_k1_DIS_s102
                                 + 0.649907 w_k1_FDX_s102
                                 + 0.733442 w_k1_IBM_s102
                                 + 0.625536 w_k1_INTC_s102
                                 + 0.578716 w_k1_KO_s102
                                 + 0.578872 w_k1_MAT_s102
                                 + 0.548543 w_k1_MSFT_s102
                                 + 0.687883 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.24221 w_k1_ADM_s102
                                 + 0.93894 w_k1_AXP_s102
                                 + 0.793574 w_k1_DIS_s102
                                 + 0.812131 w_k1_FDX_s102
                                 + 1.46731 w_k1_IBM_s102
                                 + 0.693848 w_k1_INTC_s102
                                 + 1.04256 w_k1_KO_s102 + 1.54084 w_k1_MAT_s102
                                 + 0.772769 w_k1_MSFT_s102
                                 + 0.929905 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.72095 w_k1_ADM_s102
                                 + 1.86234 w_k1_AXP_s102
                                 + 1.88326 w_k1_DIS_s102
                                 + 1.51032 w_k1_FDX_s102
                                 + 1.55308 w_k1_IBM_s102
                                 + 1.09689 w_k1_INTC_s102
                                 + 1.32204 w_k1_KO_s102 + 1.38331 w_k1_MAT_s102
                                 + 1.06136 w_k1_MSFT_s102
                                 + 1.05785 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.65272 w_k1_ADM_s102
                                 + 0.853294 w_k1_AXP_s102
                                 + 0.866425 w_k1_DIS_s102
                                 + 0.991908 w_k1_FDX_s102
                                 + 0.918338 w_k1_IBM_s102
                                 + 0.830541 w_k1_INTC_s102
                                 + 0.836926 w_k1_KO_s102
                                 + 0.864237 w_k1_MAT_s102
                                 + 0.928788 w_k1_MSFT_s102
                                 + 0.886734 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.53312 w_k1_ADM_s102
                                 + 1.24775 w_k1_AXP_s102
                                 + 1.41858 w_k1_DIS_s102
                                 + 1.27857 w_k1_FDX_s102
                                 + 1.06672 w_k1_IBM_s102
                                 + 1.03271 w_k1_INTC_s102 + 1.0638 w_k1_KO_s102
                                 + 1.14881 w_k1_MAT_s102
                                 + 1.11201 w_k1_MSFT_s102 + 1.0128 w_k1_PG_s102
                                 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.03525 w_k1_ADM_s102
                                 + 0.853521 w_k1_AXP_s102
                                 + 0.797184 w_k1_DIS_s102
                                 + 0.964807 w_k1_FDX_s102
                                 + 0.629336 w_k1_IBM_s102
                                 + 0.724694 w_k1_INTC_s102
                                 + 0.692346 w_k1_KO_s102
                                 + 0.687947 w_k1_MAT_s102
                                 + 0.844207 w_k1_MSFT_s102
                                 + 0.766876 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.00831 w_k1_ADM_s102
                                 + 1.42917 w_k1_AXP_s102 + 1.1096 w_k1_DIS_s102
                                 + 1.1317 w_k1_FDX_s102 + 1.04535 w_k1_IBM_s102
                                 + 1.08897 w_k1_INTC_s102
                                 + 0.911505 w_k1_KO_s102
                                 + 1.26714 w_k1_MAT_s102
                                 + 0.771909 w_k1_MSFT_s102
                                 + 1.01455 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.929126 w_k1_ADM_s102
                                 + 0.914746 w_k1_AXP_s102
                                 + 0.627333 w_k1_DIS_s102
                                 + 0.825403 w_k1_FDX_s102
                                 + 0.812418 w_k1_IBM_s102
                                 + 0.877207 w_k1_INTC_s102
                                 + 0.673483 w_k1_KO_s102
                                 + 0.751197 w_k1_MAT_s102
                                 + 0.721211 w_k1_MSFT_s102
                                 + 0.747837 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.800246 w_k1_ADM_s103
                                 + 0.7066 w_k1_AXP_s103
                                 + 0.614414 w_k1_DIS_s103
                                 + 0.80313 w_k1_FDX_s103
                                 + 0.702481 w_k1_IBM_s103
                                 + 0.685607 w_k1_INTC_s103
                                 + 0.635448 w_k1_KO_s103
                                 + 0.695032 w_k1_MAT_s103
                                 + 0.927804 w_k1_MSFT_s103
                                 + 0.758869 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.808719 w_k1_ADM_s103
                                 + 1.00295 w_k1_AXP_s103
                                 + 0.868644 w_k1_DIS_s103
                                 + 0.869518 w_k1_FDX_s103
                                 + 1.34966 w_k1_IBM_s103
                                 + 0.73167 w_k1_INTC_s103
                                 + 1.05953 w_k1_KO_s103 + 1.86227 w_k1_MAT_s103
                                 + 0.960133 w_k1_MSFT_s103
                                 + 1.12922 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.05642 w_k1_ADM_s103
                                 + 1.09306 w_k1_AXP_s103
                                 + 1.09433 w_k1_DIS_s103 + 1.1711 w_k1_FDX_s103
                                 + 0.904049 w_k1_IBM_s103
                                 + 0.731434 w_k1_INTC_s103
                                 + 0.984874 w_k1_KO_s103
                                 + 1.16587 w_k1_MAT_s103
                                 + 1.10736 w_k1_MSFT_s103
                                 + 0.940582 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.780464 w_k1_ADM_s103
                                 + 0.787394 w_k1_AXP_s103
                                 + 0.599014 w_k1_DIS_s103
                                 + 0.702767 w_k1_FDX_s103
                                 + 1.09308 w_k1_IBM_s103
                                 + 0.856785 w_k1_INTC_s103
                                 + 0.74148 w_k1_KO_s103
                                 + 0.684813 w_k1_MAT_s103
                                 + 0.843893 w_k1_MSFT_s103
                                 + 0.795015 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.189 w_k1_ADM_s103
                                 + 1.41269 w_k1_AXP_s103
                                 + 1.24488 w_k1_DIS_s103
                                 + 1.35537 w_k1_FDX_s103
                                 + 1.02223 w_k1_IBM_s103
                                 + 1.09629 w_k1_INTC_s103
                                 + 1.04248 w_k1_KO_s103 + 1.24043 w_k1_MAT_s103
                                 + 1.23195 w_k1_MSFT_s103
                                 + 1.22619 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.04454 w_k1_ADM_s103
                                 + 1.02751 w_k1_AXP_s103
                                 + 0.847429 w_k1_DIS_s103
                                 + 0.968755 w_k1_FDX_s103
                                 + 0.698092 w_k1_IBM_s103
                                 + 0.840163 w_k1_INTC_s103
                                 + 0.790279 w_k1_KO_s103
                                 + 1.03028 w_k1_MAT_s103
                                 + 0.945951 w_k1_MSFT_s103
                                 + 0.978433 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.763956 w_k1_ADM_s103
                                 + 1.3722 w_k1_AXP_s103 + 1.38445 w_k1_DIS_s103
                                 + 1.1121 w_k1_FDX_s103 + 1.5175 w_k1_IBM_s103
                                 + 1.14389 w_k1_INTC_s103
                                 + 1.36263 w_k1_KO_s103 + 1.78128 w_k1_MAT_s103
                                 + 1.32341 w_k1_MSFT_s103
                                 + 1.14113 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.925016 w_k1_ADM_s103
                                 + 0.886804 w_k1_AXP_s103
                                 + 0.860232 w_k1_DIS_s103
                                 + 0.886186 w_k1_FDX_s103
                                 + 0.997151 w_k1_IBM_s103
                                 + 0.897394 w_k1_INTC_s103
                                 + 0.905721 w_k1_KO_s103
                                 + 0.881709 w_k1_MAT_s103
                                 + 0.95901 w_k1_MSFT_s103
                                 + 1.01469 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.782036 w_k1_ADM_s104
                                 + 0.966576 w_k1_AXP_s104
                                 + 1.00224 w_k1_DIS_s104
                                 + 0.988706 w_k1_FDX_s104
                                 + 0.962685 w_k1_IBM_s104
                                 + 0.930002 w_k1_INTC_s104
                                 + 0.988001 w_k1_KO_s104
                                 + 0.947649 w_k1_MAT_s104
                                 + 0.840145 w_k1_MSFT_s104
                                 + 1.04698 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.932286 w_k1_ADM_s104
                                 + 0.836638 w_k1_AXP_s104
                                 + 0.901199 w_k1_DIS_s104
                                 + 0.879191 w_k1_FDX_s104
                                 + 1.3342 w_k1_IBM_s104
                                 + 0.854934 w_k1_INTC_s104
                                 + 0.995391 w_k1_KO_s104
                                 + 1.07797 w_k1_MAT_s104
                                 + 0.955827 w_k1_MSFT_s104
                                 + 1.02051 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.705225 w_k1_ADM_s104
                                 + 1.1749 w_k1_AXP_s104 + 1.38551 w_k1_DIS_s104
                                 + 1.04644 w_k1_FDX_s104
                                 + 1.54068 w_k1_IBM_s104
                                 + 1.09901 w_k1_INTC_s104
                                 + 1.25734 w_k1_KO_s104 + 1.71982 w_k1_MAT_s104
                                 + 1.01694 w_k1_MSFT_s104
                                 + 1.29209 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.12325 w_k1_ADM_s104
                                 + 1.10491 w_k1_AXP_s104
                                 + 1.12162 w_k1_DIS_s104
                                 + 1.16315 w_k1_FDX_s104
                                 + 0.85177 w_k1_IBM_s104
                                 + 1.15748 w_k1_INTC_s104
                                 + 1.01793 w_k1_KO_s104
                                 + 0.754737 w_k1_MAT_s104
                                 + 1.12298 w_k1_MSFT_s104
                                 + 1.07473 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.892704 w_k1_ADM_s104
                                 + 1.14046 w_k1_AXP_s104
                                 + 1.15793 w_k1_DIS_s104
                                 + 1.07193 w_k1_FDX_s104
                                 + 1.14779 w_k1_IBM_s104
                                 + 0.97115 w_k1_INTC_s104
                                 + 1.07944 w_k1_KO_s104 + 1.43416 w_k1_MAT_s104
                                 + 1.00837 w_k1_MSFT_s104
                                 + 1.16095 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.12786 w_k1_ADM_s104
                                 + 1.27238 w_k1_AXP_s104
                                 + 1.64766 w_k1_DIS_s104
                                 + 1.35405 w_k1_FDX_s104
                                 + 1.08453 w_k1_IBM_s104
                                 + 1.32399 w_k1_INTC_s104
                                 + 1.24142 w_k1_KO_s104 + 0.97709 w_k1_MAT_s104
                                 + 1.35715 w_k1_MSFT_s104
                                 + 1.22869 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.68846 w_k1_ADM_s104
                                 + 0.885647 w_k1_AXP_s104
                                 + 0.753635 w_k1_DIS_s104
                                 + 0.873066 w_k1_FDX_s104
                                 + 0.704662 w_k1_IBM_s104
                                 + 0.824369 w_k1_INTC_s104
                                 + 0.658283 w_k1_KO_s104
                                 + 0.627217 w_k1_MAT_s104
                                 + 0.690356 w_k1_MSFT_s104
                                 + 0.851451 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.848098 w_k1_ADM_s104
                                 + 0.633292 w_k1_AXP_s104
                                 + 0.851809 w_k1_DIS_s104
                                 + 0.752771 w_k1_FDX_s104
                                 + 0.97867 w_k1_IBM_s104
                                 + 0.86028 w_k1_INTC_s104
                                 + 1.00494 w_k1_KO_s104
                                 + 0.622926 w_k1_MAT_s104
                                 + 0.842611 w_k1_MSFT_s104
                                 + 0.918416 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.12622 w_k1_ADM_s105
                                 + 1.11834 w_k1_AXP_s105
                                 + 1.09872 w_k1_DIS_s105
                                 + 1.09747 w_k1_FDX_s105
                                 + 0.982391 w_k1_IBM_s105
                                 + 0.657853 w_k1_INTC_s105
                                 + 0.950718 w_k1_KO_s105
                                 + 1.18952 w_k1_MAT_s105
                                 + 0.90043 w_k1_MSFT_s105
                                 + 0.974404 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.976439 w_k1_ADM_s105
                                 + 1.3299 w_k1_AXP_s105
                                 + 0.973885 w_k1_DIS_s105
                                 + 1.02249 w_k1_FDX_s105
                                 + 1.21587 w_k1_IBM_s105
                                 + 0.972424 w_k1_INTC_s105
                                 + 1.05108 w_k1_KO_s105 + 1.71025 w_k1_MAT_s105
                                 + 1.05238 w_k1_MSFT_s105
                                 + 1.00178 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.806379 w_k1_ADM_s105
                                 + 0.7704 w_k1_AXP_s105
                                 + 0.710657 w_k1_DIS_s105
                                 + 0.728139 w_k1_FDX_s105
                                 + 1.0078 w_k1_IBM_s105
                                 + 0.707162 w_k1_INTC_s105
                                 + 0.812608 w_k1_KO_s105
                                 + 0.96702 w_k1_MAT_s105
                                 + 0.823157 w_k1_MSFT_s105
                                 + 0.785886 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.746225 w_k1_ADM_s105
                                 + 0.814505 w_k1_AXP_s105
                                 + 0.816516 w_k1_DIS_s105
                                 + 0.98334 w_k1_FDX_s105
                                 + 0.772653 w_k1_IBM_s105
                                 + 0.780345 w_k1_INTC_s105
                                 + 0.792353 w_k1_KO_s105
                                 + 0.683381 w_k1_MAT_s105
                                 + 0.858833 w_k1_MSFT_s105
                                 + 0.812164 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.03105 w_k1_ADM_s105
                                 + 1.15612 w_k1_AXP_s105
                                 + 1.18679 w_k1_DIS_s105
                                 + 1.05678 w_k1_FDX_s105
                                 + 1.61635 w_k1_IBM_s105
                                 + 0.711391 w_k1_INTC_s105
                                 + 1.29787 w_k1_KO_s105 + 1.7412 w_k1_MAT_s105
                                 + 0.84052 w_k1_MSFT_s105
                                 + 0.899151 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.872611 w_k1_ADM_s105
                                 + 1.11261 w_k1_AXP_s105
                                 + 0.75528 w_k1_DIS_s105
                                 + 0.893181 w_k1_FDX_s105
                                 + 1.28884 w_k1_IBM_s105
                                 + 0.683919 w_k1_INTC_s105
                                 + 0.865311 w_k1_KO_s105
                                 + 1.28186 w_k1_MAT_s105
                                 + 0.798256 w_k1_MSFT_s105
                                 + 0.893776 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.36177 w_k1_ADM_s105
                                 + 1.39075 w_k1_AXP_s105
                                 + 1.61935 w_k1_DIS_s105
                                 + 1.42784 w_k1_FDX_s105
                                 + 0.914493 w_k1_IBM_s105
                                 + 0.778827 w_k1_INTC_s105
                                 + 1.08316 w_k1_KO_s105 + 1.32914 w_k1_MAT_s105
                                 + 1.41344 w_k1_MSFT_s105
                                 + 1.10945 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.22742 w_k1_ADM_s105
                                 + 1.39916 w_k1_AXP_s105
                                 + 1.62309 w_k1_DIS_s105 + 1.3487 w_k1_FDX_s105
                                 + 1.471 w_k1_IBM_s105 + 1.01433 w_k1_INTC_s105
                                 + 1.52924 w_k1_KO_s105 + 1.7064 w_k1_MAT_s105
                                 + 1.20992 w_k1_MSFT_s105
                                 + 1.22878 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.14783 w_k1_ADM_s106
                                 + 1.05911 w_k1_AXP_s106
                                 + 1.03063 w_k1_DIS_s106
                                 + 1.03649 w_k1_FDX_s106
                                 + 0.905522 w_k1_IBM_s106
                                 + 1.10964 w_k1_INTC_s106
                                 + 0.930485 w_k1_KO_s106
                                 + 0.792192 w_k1_MAT_s106
                                 + 1.21168 w_k1_MSFT_s106
                                 + 0.952303 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.08795 w_k1_ADM_s106
                                 + 0.863681 w_k1_AXP_s106
                                 + 0.789449 w_k1_DIS_s106
                                 + 0.996963 w_k1_FDX_s106
                                 + 0.671607 w_k1_IBM_s106
                                 + 0.874665 w_k1_INTC_s106
                                 + 0.788903 w_k1_KO_s106
                                 + 0.660983 w_k1_MAT_s106
                                 + 0.964727 w_k1_MSFT_s106
                                 + 0.830014 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.27877 w_k1_ADM_s106
                                 + 1.23577 w_k1_AXP_s106
                                 + 1.67311 w_k1_DIS_s106
                                 + 1.24374 w_k1_FDX_s106
                                 + 1.37603 w_k1_IBM_s106
                                 + 1.47465 w_k1_INTC_s106
                                 + 1.62644 w_k1_KO_s106 + 1.29382 w_k1_MAT_s106
                                 + 1.46672 w_k1_MSFT_s106
                                 + 1.21574 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.05472 w_k1_ADM_s106
                                 + 1.17801 w_k1_AXP_s106
                                 + 1.10483 w_k1_DIS_s106
                                 + 1.08403 w_k1_FDX_s106
                                 + 1.16066 w_k1_IBM_s106
                                 + 0.893499 w_k1_INTC_s106
                                 + 1.12101 w_k1_KO_s106 + 1.01509 w_k1_MAT_s106
                                 + 0.874348 w_k1_MSFT_s106
                                 + 1.08609 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.712843 w_k1_ADM_s106
                                 + 0.763167 w_k1_AXP_s106
                                 + 0.883256 w_k1_DIS_s106
                                 + 0.757591 w_k1_FDX_s106
                                 + 1.22812 w_k1_IBM_s106
                                 + 0.969617 w_k1_INTC_s106
                                 + 0.859243 w_k1_KO_s106
                                 + 0.854219 w_k1_MAT_s106
                                 + 0.9008 w_k1_MSFT_s106
                                 + 0.830039 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.05074 w_k1_ADM_s106
                                 + 1.36671 w_k1_AXP_s106 + 1.132 w_k1_DIS_s106
                                 + 1.02416 w_k1_FDX_s106
                                 + 1.84217 w_k1_IBM_s106
                                 + 1.14532 w_k1_INTC_s106
                                 + 1.42715 w_k1_KO_s106 + 1.8177 w_k1_MAT_s106
                                 + 1.16238 w_k1_MSFT_s106
                                 + 1.09641 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.868434 w_k1_ADM_s106
                                 + 0.876238 w_k1_AXP_s106
                                 + 0.798189 w_k1_DIS_s106
                                 + 0.843424 w_k1_FDX_s106
                                 + 1.15786 w_k1_IBM_s106
                                 + 0.882551 w_k1_INTC_s106
                                 + 0.928655 w_k1_KO_s106
                                 + 1.01393 w_k1_MAT_s106
                                 + 0.838726 w_k1_MSFT_s106
                                 + 0.98431 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.19387 w_k1_ADM_s106
                                 + 1.615 w_k1_AXP_s106 + 1.40324 w_k1_DIS_s106
                                 + 1.28765 w_k1_FDX_s106
                                 + 0.824654 w_k1_IBM_s106
                                 + 1.15433 w_k1_INTC_s106
                                 + 1.00719 w_k1_KO_s106 + 0.9581 w_k1_MAT_s106
                                 + 1.11682 w_k1_MSFT_s106
                                 + 1.03444 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.741109 w_k1_ADM_s107
                                 + 1.12656 w_k1_AXP_s107
                                 + 0.813047 w_k1_DIS_s107
                                 + 0.973001 w_k1_FDX_s107
                                 + 0.94335 w_k1_IBM_s107
                                 + 0.77139 w_k1_INTC_s107
                                 + 0.90544 w_k1_KO_s107 + 1.06543 w_k1_MAT_s107
                                 + 0.704885 w_k1_MSFT_s107
                                 + 0.935911 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.01093 w_k1_ADM_s107
                                 + 1.74303 w_k1_AXP_s107
                                 + 2.07917 w_k1_DIS_s107
                                 + 1.45636 w_k1_FDX_s107
                                 + 1.34614 w_k1_IBM_s107
                                 + 1.24221 w_k1_INTC_s107
                                 + 1.59242 w_k1_KO_s107 + 1.68611 w_k1_MAT_s107
                                 + 1.35026 w_k1_MSFT_s107
                                 + 1.05349 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.606171 w_k1_ADM_s107
                                 + 0.888503 w_k1_AXP_s107
                                 + 1.02276 w_k1_DIS_s107
                                 + 0.938396 w_k1_FDX_s107
                                 + 1.04026 w_k1_IBM_s107
                                 + 0.893187 w_k1_INTC_s107
                                 + 1.01198 w_k1_KO_s107 + 1.02888 w_k1_MAT_s107
                                 + 0.762185 w_k1_MSFT_s107
                                 + 0.876668 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.986422 w_k1_ADM_s107
                                 + 1.38091 w_k1_AXP_s107
                                 + 1.36181 w_k1_DIS_s107
                                 + 1.10693 w_k1_FDX_s107
                                 + 1.41224 w_k1_IBM_s107
                                 + 0.844673 w_k1_INTC_s107
                                 + 1.45239 w_k1_KO_s107 + 1.55014 w_k1_MAT_s107
                                 + 0.925114 w_k1_MSFT_s107
                                 + 1.04002 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.649219 w_k1_ADM_s107
                                 + 0.814282 w_k1_AXP_s107
                                 + 0.657456 w_k1_DIS_s107
                                 + 0.746003 w_k1_FDX_s107
                                 + 0.701886 w_k1_IBM_s107
                                 + 0.610492 w_k1_INTC_s107
                                 + 0.782234 w_k1_KO_s107
                                 + 0.813445 w_k1_MAT_s107
                                 + 0.675349 w_k1_MSFT_s107
                                 + 0.755628 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.806116 w_k1_ADM_s107
                                 + 0.841677 w_k1_AXP_s107
                                 + 1.01935 w_k1_DIS_s107
                                 + 1.04791 w_k1_FDX_s107
                                 + 0.750151 w_k1_IBM_s107
                                 + 1.16056 w_k1_INTC_s107
                                 + 1.08843 w_k1_KO_s107
                                 + 0.786849 w_k1_MAT_s107
                                 + 1.03696 w_k1_MSFT_s107
                                 + 0.856236 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.613778 w_k1_ADM_s107
                                 + 0.864404 w_k1_AXP_s107
                                 + 0.912062 w_k1_DIS_s107
                                 + 0.912037 w_k1_FDX_s107
                                 + 1.30494 w_k1_IBM_s107
                                 + 0.859093 w_k1_INTC_s107
                                 + 1.21348 w_k1_KO_s107 + 1.45161 w_k1_MAT_s107
                                 + 0.831342 w_k1_MSFT_s107
                                 + 0.917961 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.910363 w_k1_ADM_s107
                                 + 1.28246 w_k1_AXP_s107
                                 + 0.95264 w_k1_DIS_s107
                                 + 1.25077 w_k1_FDX_s107
                                 + 0.774882 w_k1_IBM_s107
                                 + 1.27618 w_k1_INTC_s107
                                 + 0.905181 w_k1_KO_s107
                                 + 0.846913 w_k1_MAT_s107
                                 + 1.02159 w_k1_MSFT_s107
                                 + 1.00281 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.15582 w_k1_ADM_s110
                                 + 1.09713 w_k1_AXP_s110
                                 + 1.20545 w_k1_DIS_s110
                                 + 1.23572 w_k1_FDX_s110
                                 + 0.762295 w_k1_IBM_s110
                                 + 1.23765 w_k1_INTC_s110
                                 + 0.946571 w_k1_KO_s110
                                 + 0.690697 w_k1_MAT_s110
                                 + 1.29477 w_k1_MSFT_s110
                                 + 1.02599 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.749349 w_k1_ADM_s110
                                 + 1.00893 w_k1_AXP_s110
                                 + 0.726913 w_k1_DIS_s110
                                 + 0.837631 w_k1_FDX_s110
                                 + 1.01955 w_k1_IBM_s110
                                 + 0.910178 w_k1_INTC_s110
                                 + 0.862871 w_k1_KO_s110
                                 + 0.940388 w_k1_MAT_s110
                                 + 0.818707 w_k1_MSFT_s110
                                 + 0.902136 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.882756 w_k1_ADM_s110
                                 + 0.844817 w_k1_AXP_s110
                                 + 0.779422 w_k1_DIS_s110
                                 + 0.940626 w_k1_FDX_s110
                                 + 0.758361 w_k1_IBM_s110
                                 + 1.03229 w_k1_INTC_s110
                                 + 0.844992 w_k1_KO_s110
                                 + 0.368722 w_k1_MAT_s110
                                 + 0.941697 w_k1_MSFT_s110
                                 + 0.894702 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.629007 w_k1_ADM_s110
                                 + 0.726506 w_k1_AXP_s110
                                 + 0.583164 w_k1_DIS_s110
                                 + 0.668408 w_k1_FDX_s110
                                 + 0.78616 w_k1_IBM_s110
                                 + 0.839257 w_k1_INTC_s110
                                 + 0.761264 w_k1_KO_s110
                                 + 0.679159 w_k1_MAT_s110
                                 + 0.691375 w_k1_MSFT_s110
                                 + 0.85164 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.673388 w_k1_ADM_s110
                                 + 0.95875 w_k1_AXP_s110
                                 + 0.96253 w_k1_DIS_s110
                                 + 0.947321 w_k1_FDX_s110
                                 + 0.865777 w_k1_IBM_s110
                                 + 1.16771 w_k1_INTC_s110
                                 + 0.969587 w_k1_KO_s110
                                 + 1.10918 w_k1_MAT_s110
                                 + 0.953484 w_k1_MSFT_s110
                                 + 1.02505 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.20545 w_k1_ADM_s110
                                 + 1.47289 w_k1_AXP_s110
                                 + 1.76156 w_k1_DIS_s110
                                 + 1.39375 w_k1_FDX_s110
                                 + 1.21377 w_k1_IBM_s110
                                 + 1.45605 w_k1_INTC_s110
                                 + 1.37397 w_k1_KO_s110
                                 + 0.985137 w_k1_MAT_s110
                                 + 1.59939 w_k1_MSFT_s110
                                 + 1.40968 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.00832 w_k1_ADM_s110
                                 + 1.13394 w_k1_AXP_s110
                                 + 1.00901 w_k1_DIS_s110
                                 + 0.935213 w_k1_FDX_s110
                                 + 1.44437 w_k1_IBM_s110
                                 + 0.992353 w_k1_INTC_s110
                                 + 1.2538 w_k1_KO_s110 + 0.89164 w_k1_MAT_s110
                                 + 0.8708 w_k1_MSFT_s110 + 1.05783 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.06655 w_k1_ADM_s110
                                 + 1.25745 w_k1_AXP_s110
                                 + 1.21633 w_k1_DIS_s110 + 1.098 w_k1_FDX_s110
                                 + 1.15001 w_k1_IBM_s110
                                 + 1.25446 w_k1_INTC_s110
                                 + 1.10595 w_k1_KO_s110
                                 + 0.702251 w_k1_MAT_s110
                                 + 1.17893 w_k1_MSFT_s110
                                 + 1.20718 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.843125 w_k1_ADM_s111
                                 + 0.756067 w_k1_AXP_s111
                                 + 0.754868 w_k1_DIS_s111
                                 + 0.871081 w_k1_FDX_s111
                                 + 0.949986 w_k1_IBM_s111
                                 + 0.86067 w_k1_INTC_s111
                                 + 0.968651 w_k1_KO_s111
                                 + 1.22031 w_k1_MAT_s111
                                 + 0.837821 w_k1_MSFT_s111
                                 + 1.04167 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.19128 w_k1_ADM_s111
                                 + 1.11243 w_k1_AXP_s111
                                 + 1.02422 w_k1_DIS_s111
                                 + 1.07007 w_k1_FDX_s111
                                 + 1.22209 w_k1_IBM_s111
                                 + 1.11119 w_k1_INTC_s111
                                 + 1.32007 w_k1_KO_s111 + 1.44267 w_k1_MAT_s111
                                 + 0.970961 w_k1_MSFT_s111
                                 + 1.04177 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.27527 w_k1_ADM_s111
                                 + 1.36597 w_k1_AXP_s111
                                 + 1.39251 w_k1_DIS_s111
                                 + 1.25657 w_k1_FDX_s111
                                 + 0.898999 w_k1_IBM_s111
                                 + 1.11482 w_k1_INTC_s111
                                 + 1.34678 w_k1_KO_s111 + 1.31407 w_k1_MAT_s111
                                 + 1.10101 w_k1_MSFT_s111
                                 + 1.09547 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.994583 w_k1_ADM_s111
                                 + 1.21803 w_k1_AXP_s111
                                 + 1.56848 w_k1_DIS_s111
                                 + 1.18467 w_k1_FDX_s111
                                 + 1.38608 w_k1_IBM_s111
                                 + 1.27684 w_k1_INTC_s111
                                 + 1.69148 w_k1_KO_s111 + 1.22357 w_k1_MAT_s111
                                 + 1.1511 w_k1_MSFT_s111 + 1.12289 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.757847 w_k1_ADM_s111
                                 + 0.671686 w_k1_AXP_s111
                                 + 0.680353 w_k1_DIS_s111
                                 + 0.741613 w_k1_FDX_s111
                                 + 0.742026 w_k1_IBM_s111
                                 + 0.867129 w_k1_INTC_s111
                                 + 0.804594 w_k1_KO_s111
                                 + 1.02611 w_k1_MAT_s111
                                 + 0.777363 w_k1_MSFT_s111
                                 + 0.851485 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.11019 w_k1_ADM_s111
                                 + 0.937942 w_k1_AXP_s111
                                 + 0.817728 w_k1_DIS_s111
                                 + 1.02479 w_k1_FDX_s111
                                 + 0.665171 w_k1_IBM_s111
                                 + 0.952028 w_k1_INTC_s111
                                 + 0.919996 w_k1_KO_s111
                                 + 0.663749 w_k1_MAT_s111
                                 + 0.944704 w_k1_MSFT_s111
                                 + 0.898047 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.08309 w_k1_ADM_s111
                                 + 0.845633 w_k1_AXP_s111
                                 + 0.972941 w_k1_DIS_s111
                                 + 0.995664 w_k1_FDX_s111
                                 + 0.844601 w_k1_IBM_s111
                                 + 0.868158 w_k1_INTC_s111
                                 + 0.97072 w_k1_KO_s111
                                 + 0.927358 w_k1_MAT_s111
                                 + 0.916075 w_k1_MSFT_s111
                                 + 0.924909 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.798471 w_k1_ADM_s111
                                 + 0.982113 w_k1_AXP_s111
                                 + 0.912669 w_k1_DIS_s111
                                 + 0.851425 w_k1_FDX_s111
                                 + 1.17767 w_k1_IBM_s111
                                 + 0.92058 w_k1_INTC_s111
                                 + 1.20969 w_k1_KO_s111 + 1.41489 w_k1_MAT_s111
                                 + 0.867895 w_k1_MSFT_s111
                                 + 1.0235 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.07759 w_k1_ADM_s112
                                 + 1.11184 w_k1_AXP_s112
                                 + 0.993511 w_k1_DIS_s112
                                 + 1.1401 w_k1_FDX_s112
                                 + 0.905754 w_k1_IBM_s112
                                 + 1.09531 w_k1_INTC_s112
                                 + 0.974954 w_k1_KO_s112
                                 + 1.03533 w_k1_MAT_s112
                                 + 0.95134 w_k1_MSFT_s112
                                 + 1.12574 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.08144 w_k1_ADM_s112
                                 + 1.53531 w_k1_AXP_s112
                                 + 1.51164 w_k1_DIS_s112
                                 + 1.30356 w_k1_FDX_s112
                                 + 1.63798 w_k1_IBM_s112
                                 + 0.951026 w_k1_INTC_s112
                                 + 1.1968 w_k1_KO_s112 + 2.02449 w_k1_MAT_s112
                                 + 0.911135 w_k1_MSFT_s112
                                 + 1.21596 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.942447 w_k1_ADM_s112
                                 + 1.1896 w_k1_AXP_s112 + 1.05911 w_k1_DIS_s112
                                 + 1.0523 w_k1_FDX_s112 + 1.02928 w_k1_IBM_s112
                                 + 1.06085 w_k1_INTC_s112
                                 + 1.05555 w_k1_KO_s112 + 1.3132 w_k1_MAT_s112
                                 + 0.876013 w_k1_MSFT_s112
                                 + 1.0256 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.12699 w_k1_ADM_s112
                                 + 1.26984 w_k1_AXP_s112 + 1.1959 w_k1_DIS_s112
                                 + 1.23397 w_k1_FDX_s112
                                 + 1.29088 w_k1_IBM_s112
                                 + 1.19027 w_k1_INTC_s112
                                 + 1.17927 w_k1_KO_s112 + 1.49245 w_k1_MAT_s112
                                 + 1.0403 w_k1_MSFT_s112 + 1.16707 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.772008 w_k1_ADM_s112
                                 + 0.759703 w_k1_AXP_s112
                                 + 0.72436 w_k1_DIS_s112
                                 + 0.81218 w_k1_FDX_s112
                                 + 0.772066 w_k1_IBM_s112
                                 + 0.754265 w_k1_INTC_s112
                                 + 0.844112 w_k1_KO_s112
                                 + 0.813906 w_k1_MAT_s112
                                 + 0.693899 w_k1_MSFT_s112
                                 + 0.989823 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.08837 w_k1_ADM_s112
                                 + 1.0942 w_k1_AXP_s112 + 1.63599 w_k1_DIS_s112
                                 + 1.2091 w_k1_FDX_s112 + 1.03014 w_k1_IBM_s112
                                 + 1.04109 w_k1_INTC_s112
                                 + 1.23216 w_k1_KO_s112 + 1.02196 w_k1_MAT_s112
                                 + 1.18243 w_k1_MSFT_s112
                                 + 1.18801 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.913217 w_k1_ADM_s112
                                 + 0.824315 w_k1_AXP_s112
                                 + 0.91079 w_k1_DIS_s112
                                 + 0.886129 w_k1_FDX_s112
                                 + 1.05274 w_k1_IBM_s112
                                 + 0.850449 w_k1_INTC_s112
                                 + 1.03809 w_k1_KO_s112 + 1.10798 w_k1_MAT_s112
                                 + 0.942694 w_k1_MSFT_s112
                                 + 0.954408 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.12326 w_k1_ADM_s112
                                 + 1.18765 w_k1_AXP_s112
                                 + 1.05834 w_k1_DIS_s112
                                 + 1.17136 w_k1_FDX_s112
                                 + 0.768076 w_k1_IBM_s112
                                 + 1.04612 w_k1_INTC_s112
                                 + 0.882044 w_k1_KO_s112
                                 + 0.793485 w_k1_MAT_s112
                                 + 1.03126 w_k1_MSFT_s112
                                 + 1.07291 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.10116 w_k1_ADM_s113
                                 + 0.726192 w_k1_AXP_s113
                                 + 0.726086 w_k1_DIS_s113
                                 + 0.863817 w_k1_FDX_s113
                                 + 0.58821 w_k1_IBM_s113
                                 + 0.642262 w_k1_INTC_s113
                                 + 0.663524 w_k1_KO_s113
                                 + 0.476334 w_k1_MAT_s113
                                 + 0.811729 w_k1_MSFT_s113
                                 + 0.857124 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.20412 w_k1_ADM_s113
                                 + 1.05928 w_k1_AXP_s113
                                 + 1.48471 w_k1_DIS_s113
                                 + 1.16961 w_k1_FDX_s113
                                 + 0.990261 w_k1_IBM_s113
                                 + 1.02989 w_k1_INTC_s113
                                 + 1.08565 w_k1_KO_s113 + 1.18092 w_k1_MAT_s113
                                 + 1.20789 w_k1_MSFT_s113
                                 + 1.15576 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.11736 w_k1_ADM_s113
                                 + 0.833637 w_k1_AXP_s113
                                 + 1.14628 w_k1_DIS_s113
                                 + 0.916276 w_k1_FDX_s113
                                 + 1.00655 w_k1_IBM_s113
                                 + 0.786275 w_k1_INTC_s113
                                 + 0.980176 w_k1_KO_s113
                                 + 0.788823 w_k1_MAT_s113
                                 + 0.957646 w_k1_MSFT_s113
                                 + 1.06187 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.0201 w_k1_ADM_s113
                                 + 0.648553 w_k1_AXP_s113
                                 + 0.768234 w_k1_DIS_s113
                                 + 0.810278 w_k1_FDX_s113
                                 + 1.0454 w_k1_IBM_s113
                                 + 0.677253 w_k1_INTC_s113
                                 + 0.894302 w_k1_KO_s113
                                 + 0.81565 w_k1_MAT_s113
                                 + 0.899385 w_k1_MSFT_s113
                                 + 0.955861 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.18423 w_k1_ADM_s113
                                 + 1.0527 w_k1_AXP_s113 + 1.14028 w_k1_DIS_s113
                                 + 0.941866 w_k1_FDX_s113
                                 + 1.54009 w_k1_IBM_s113
                                 + 0.860009 w_k1_INTC_s113
                                 + 1.29118 w_k1_KO_s113 + 1.82634 w_k1_MAT_s113
                                 + 1.0367 w_k1_MSFT_s113 + 1.14744 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.09906 w_k1_ADM_s113
                                 + 1.137 w_k1_AXP_s113 + 0.897203 w_k1_DIS_s113
                                 + 0.961596 w_k1_FDX_s113
                                 + 0.914518 w_k1_IBM_s113
                                 + 0.891334 w_k1_INTC_s113
                                 + 0.81445 w_k1_KO_s113 + 1.0194 w_k1_MAT_s113
                                 + 0.898761 w_k1_MSFT_s113
                                 + 0.99369 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.43124 w_k1_ADM_s113
                                 + 1.61163 w_k1_AXP_s113
                                 + 1.42212 w_k1_DIS_s113
                                 + 1.32988 w_k1_FDX_s113
                                 + 0.947732 w_k1_IBM_s113
                                 + 0.994583 w_k1_INTC_s113
                                 + 1.01204 w_k1_KO_s113 + 1.14774 w_k1_MAT_s113
                                 + 1.04634 w_k1_MSFT_s113
                                 + 1.14744 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.944768 w_k1_ADM_s113
                                 + 0.676818 w_k1_AXP_s113
                                 + 0.715962 w_k1_DIS_s113
                                 + 0.753425 w_k1_FDX_s113
                                 + 0.750681 w_k1_IBM_s113
                                 + 0.813878 w_k1_INTC_s113
                                 + 0.831873 w_k1_KO_s113
                                 + 0.806461 w_k1_MAT_s113
                                 + 0.885852 w_k1_MSFT_s113
                                 + 0.943449 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.09207 w_k1_ADM_s114
                                 + 1.23854 w_k1_AXP_s114
                                 + 1.10938 w_k1_DIS_s114
                                 + 1.06796 w_k1_FDX_s114
                                 + 1.16152 w_k1_IBM_s114
                                 + 1.35925 w_k1_INTC_s114
                                 + 1.13888 w_k1_KO_s114 + 1.27552 w_k1_MAT_s114
                                 + 1.04072 w_k1_MSFT_s114
                                 + 1.03063 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.807812 w_k1_ADM_s114
                                 + 0.783869 w_k1_AXP_s114
                                 + 0.681214 w_k1_DIS_s114
                                 + 0.753013 w_k1_FDX_s114
                                 + 0.720063 w_k1_IBM_s114
                                 + 0.939483 w_k1_INTC_s114
                                 + 0.751589 w_k1_KO_s114
                                 + 0.537942 w_k1_MAT_s114
                                 + 0.664911 w_k1_MSFT_s114
                                 + 0.821727 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.13684 w_k1_ADM_s114
                                 + 1.09795 w_k1_AXP_s114
                                 + 1.55854 w_k1_DIS_s114
                                 + 1.26572 w_k1_FDX_s114
                                 + 0.970073 w_k1_IBM_s114
                                 + 1.45189 w_k1_INTC_s114
                                 + 1.07156 w_k1_KO_s114 + 0.67598 w_k1_MAT_s114
                                 + 1.15948 w_k1_MSFT_s114
                                 + 1.11764 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.845815 w_k1_ADM_s114
                                 + 0.893848 w_k1_AXP_s114
                                 + 0.704572 w_k1_DIS_s114
                                 + 0.977502 w_k1_FDX_s114
                                 + 0.596268 w_k1_IBM_s114
                                 + 0.848776 w_k1_INTC_s114
                                 + 0.659761 w_k1_KO_s114
                                 + 0.702614 w_k1_MAT_s114
                                 + 0.889992 w_k1_MSFT_s114
                                 + 0.87302 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.767946 w_k1_ADM_s114
                                 + 0.902061 w_k1_AXP_s114
                                 + 0.938538 w_k1_DIS_s114
                                 + 0.887297 w_k1_FDX_s114
                                 + 1.0129 w_k1_IBM_s114
                                 + 0.90048 w_k1_INTC_s114
                                 + 0.959273 w_k1_KO_s114
                                 + 0.94756 w_k1_MAT_s114
                                 + 0.953563 w_k1_MSFT_s114
                                 + 0.91048 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.832378 w_k1_ADM_s114
                                 + 1.34975 w_k1_AXP_s114
                                 + 1.22427 w_k1_DIS_s114
                                 + 1.04209 w_k1_FDX_s114
                                 + 1.61224 w_k1_IBM_s114
                                 + 1.29111 w_k1_INTC_s114
                                 + 1.38552 w_k1_KO_s114 + 1.74281 w_k1_MAT_s114
                                 + 1.02293 w_k1_MSFT_s114
                                 + 1.13421 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.09047 w_k1_ADM_s114
                                 + 1.24641 w_k1_AXP_s114
                                 + 1.08036 w_k1_DIS_s114
                                 + 1.15817 w_k1_FDX_s114
                                 + 0.741682 w_k1_IBM_s114
                                 + 1.12242 w_k1_INTC_s114
                                 + 0.834019 w_k1_KO_s114
                                 + 0.942661 w_k1_MAT_s114
                                 + 1.00718 w_k1_MSFT_s114
                                 + 0.932824 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.30712 w_k1_ADM_s114
                                 + 1.49779 w_k1_AXP_s114
                                 + 1.78609 w_k1_DIS_s114
                                 + 1.39051 w_k1_FDX_s114
                                 + 1.40415 w_k1_IBM_s114
                                 + 1.07974 w_k1_INTC_s114
                                 + 1.44881 w_k1_KO_s114 + 2.05807 w_k1_MAT_s114
                                 + 1.22985 w_k1_MSFT_s114
                                 + 1.20535 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.17881 w_k1_ADM_s115
                                 + 1.07303 w_k1_AXP_s115
                                 + 0.950843 w_k1_DIS_s115
                                 + 1.10375 w_k1_FDX_s115
                                 + 1.19703 w_k1_IBM_s115
                                 + 1.12188 w_k1_INTC_s115
                                 + 1.15834 w_k1_KO_s115 + 1.30338 w_k1_MAT_s115
                                 + 0.985356 w_k1_MSFT_s115
                                 + 1.0511 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.04413 w_k1_ADM_s115
                                 + 1.17859 w_k1_AXP_s115
                                 + 1.21826 w_k1_DIS_s115
                                 + 1.02538 w_k1_FDX_s115 + 1.6331 w_k1_IBM_s115
                                 + 1.02376 w_k1_INTC_s115
                                 + 1.40501 w_k1_KO_s115 + 1.32858 w_k1_MAT_s115
                                 + 0.844899 w_k1_MSFT_s115
                                 + 1.16173 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.782914 w_k1_ADM_s115
                                 + 0.855681 w_k1_AXP_s115
                                 + 0.674861 w_k1_DIS_s115
                                 + 0.833936 w_k1_FDX_s115
                                 + 1.03779 w_k1_IBM_s115
                                 + 0.935711 w_k1_INTC_s115
                                 + 0.821032 w_k1_KO_s115
                                 + 0.872326 w_k1_MAT_s115
                                 + 0.771449 w_k1_MSFT_s115
                                 + 0.772091 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.21646 w_k1_ADM_s115
                                 + 1.2799 w_k1_AXP_s115 + 1.28889 w_k1_DIS_s115
                                 + 1.25609 w_k1_FDX_s115
                                 + 0.918852 w_k1_IBM_s115
                                 + 1.33135 w_k1_INTC_s115
                                 + 1.21715 w_k1_KO_s115
                                 + 0.896487 w_k1_MAT_s115
                                 + 0.978739 w_k1_MSFT_s115
                                 + 1.15888 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.792009 w_k1_ADM_s115
                                 + 0.833946 w_k1_AXP_s115
                                 + 0.759099 w_k1_DIS_s115
                                 + 0.981573 w_k1_FDX_s115
                                 + 0.640556 w_k1_IBM_s115
                                 + 1.07242 w_k1_INTC_s115
                                 + 0.859115 w_k1_KO_s115
                                 + 0.476139 w_k1_MAT_s115
                                 + 0.854162 w_k1_MSFT_s115
                                 + 0.787029 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.3391 w_k1_ADM_s115
                                 + 2.0786 w_k1_AXP_s115 + 1.92453 w_k1_DIS_s115
                                 + 1.70739 w_k1_FDX_s115
                                 + 1.39544 w_k1_IBM_s115
                                 + 1.56974 w_k1_INTC_s115
                                 + 1.42816 w_k1_KO_s115 + 1.92354 w_k1_MAT_s115
                                 + 1.4272 w_k1_MSFT_s115 + 1.43733 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.1419 w_k1_ADM_s115
                                 + 1.01798 w_k1_AXP_s115
                                 + 1.28728 w_k1_DIS_s115
                                 + 1.13692 w_k1_FDX_s115
                                 + 1.13267 w_k1_IBM_s115
                                 + 1.11372 w_k1_INTC_s115
                                 + 1.14009 w_k1_KO_s115 + 1.0121 w_k1_MAT_s115
                                 + 1.03223 w_k1_MSFT_s115
                                 + 1.19876 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.952853 w_k1_ADM_s115
                                 + 0.794697 w_k1_AXP_s115
                                 + 0.707128 w_k1_DIS_s115
                                 + 0.800315 w_k1_FDX_s115
                                 + 0.843004 w_k1_IBM_s115
                                 + 0.832203 w_k1_INTC_s115
                                 + 0.906152 w_k1_KO_s115
                                 + 0.902134 w_k1_MAT_s115
                                 + 0.761644 w_k1_MSFT_s115
                                 + 0.875818 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.09648 w_k1_ADM_s116
                                 + 1.12511 w_k1_AXP_s116
                                 + 1.04689 w_k1_DIS_s116 + 1.1125 w_k1_FDX_s116
                                 + 0.832329 w_k1_IBM_s116
                                 + 0.954009 w_k1_INTC_s116
                                 + 0.821081 w_k1_KO_s116
                                 + 0.74946 w_k1_MAT_s116
                                 + 0.935052 w_k1_MSFT_s116
                                 + 1.03256 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.881239 w_k1_ADM_s116
                                 + 1.06601 w_k1_AXP_s116
                                 + 0.985571 w_k1_DIS_s116
                                 + 1.03005 w_k1_FDX_s116
                                 + 0.592187 w_k1_IBM_s116
                                 + 0.873432 w_k1_INTC_s116
                                 + 0.683445 w_k1_KO_s116
                                 + 0.612779 w_k1_MAT_s116
                                 + 0.849365 w_k1_MSFT_s116
                                 + 0.792923 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.24995 w_k1_ADM_s116
                                 + 1.41184 w_k1_AXP_s116
                                 + 1.36097 w_k1_DIS_s116
                                 + 1.04702 w_k1_FDX_s116
                                 + 1.70121 w_k1_IBM_s116
                                 + 1.24073 w_k1_INTC_s116
                                 + 1.29708 w_k1_KO_s116 + 1.42955 w_k1_MAT_s116
                                 + 1.11653 w_k1_MSFT_s116
                                 + 1.20402 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.851561 w_k1_ADM_s116
                                 + 0.760129 w_k1_AXP_s116
                                 + 0.758529 w_k1_DIS_s116
                                 + 0.866915 w_k1_FDX_s116
                                 + 0.739968 w_k1_IBM_s116
                                 + 0.851768 w_k1_INTC_s116
                                 + 0.7222 w_k1_KO_s116 + 0.609323 w_k1_MAT_s116
                                 + 0.795427 w_k1_MSFT_s116
                                 + 0.834496 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.03383 w_k1_ADM_s116
                                 + 1.21544 w_k1_AXP_s116
                                 + 1.14598 w_k1_DIS_s116
                                 + 1.09366 w_k1_FDX_s116
                                 + 1.30652 w_k1_IBM_s116
                                 + 1.02043 w_k1_INTC_s116
                                 + 1.02829 w_k1_KO_s116 + 1.28141 w_k1_MAT_s116
                                 + 0.876232 w_k1_MSFT_s116
                                 + 1.10026 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.06184 w_k1_ADM_s116
                                 + 1.2692 w_k1_AXP_s116 + 1.04527 w_k1_DIS_s116
                                 + 1.118 w_k1_FDX_s116 + 0.999904 w_k1_IBM_s116
                                 + 1.10049 w_k1_INTC_s116
                                 + 0.950247 w_k1_KO_s116 + 1.0596 w_k1_MAT_s116
                                 + 0.998349 w_k1_MSFT_s116
                                 + 1.0349 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.32533 w_k1_ADM_s116
                                 + 1.49465 w_k1_AXP_s116
                                 + 1.73372 w_k1_DIS_s116
                                 + 1.56224 w_k1_FDX_s116 + 1.106 w_k1_IBM_s116
                                 + 1.31024 w_k1_INTC_s116
                                 + 1.12659 w_k1_KO_s116 + 1.08551 w_k1_MAT_s116
                                 + 1.33947 w_k1_MSFT_s116
                                 + 1.34428 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.673631 w_k1_ADM_s116
                                 + 0.817471 w_k1_AXP_s116
                                 + 0.665784 w_k1_DIS_s116
                                 + 0.770805 w_k1_FDX_s116
                                 + 1.00353 w_k1_IBM_s116
                                 + 0.879295 w_k1_INTC_s116
                                 + 0.748965 w_k1_KO_s116
                                 + 0.969383 w_k1_MAT_s116
                                 + 0.751234 w_k1_MSFT_s116
                                 + 0.84965 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.899231 w_k1_ADM_s117
                                 + 0.913049 w_k1_AXP_s117
                                 + 1.10457 w_k1_DIS_s117
                                 + 0.986528 w_k1_FDX_s117
                                 + 1.03129 w_k1_IBM_s117
                                 + 1.04367 w_k1_INTC_s117
                                 + 0.996729 w_k1_KO_s117
                                 + 0.894768 w_k1_MAT_s117
                                 + 1.02069 w_k1_MSFT_s117
                                 + 0.982763 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.899694 w_k1_ADM_s117
                                 + 1.08536 w_k1_AXP_s117
                                 + 0.660187 w_k1_DIS_s117
                                 + 0.843069 w_k1_FDX_s117
                                 + 1.08551 w_k1_IBM_s117
                                 + 0.879813 w_k1_INTC_s117
                                 + 1.06258 w_k1_KO_s117 + 1.42876 w_k1_MAT_s117
                                 + 0.666817 w_k1_MSFT_s117
                                 + 0.925635 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.19094 w_k1_ADM_s117
                                 + 1.10366 w_k1_AXP_s117
                                 + 1.32169 w_k1_DIS_s117
                                 + 1.09865 w_k1_FDX_s117
                                 + 1.35044 w_k1_IBM_s117
                                 + 1.16435 w_k1_INTC_s117
                                 + 1.41689 w_k1_KO_s117 + 1.21549 w_k1_MAT_s117
                                 + 1.02313 w_k1_MSFT_s117
                                 + 1.06565 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.921894 w_k1_ADM_s117
                                 + 0.977654 w_k1_AXP_s117
                                 + 1.0266 w_k1_DIS_s117 + 1.03355 w_k1_FDX_s117
                                 + 0.790251 w_k1_IBM_s117
                                 + 1.07499 w_k1_INTC_s117
                                 + 1.05357 w_k1_KO_s117
                                 + 0.955403 w_k1_MAT_s117
                                 + 0.972805 w_k1_MSFT_s117
                                 + 0.987453 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.602459 w_k1_ADM_s117
                                 + 0.722176 w_k1_AXP_s117
                                 + 0.711128 w_k1_DIS_s117
                                 + 0.794144 w_k1_FDX_s117
                                 + 0.981381 w_k1_IBM_s117
                                 + 1.00559 w_k1_INTC_s117
                                 + 0.861722 w_k1_KO_s117
                                 + 0.777283 w_k1_MAT_s117
                                 + 0.804095 w_k1_MSFT_s117
                                 + 0.797262 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.783886 w_k1_ADM_s117
                                 + 1.16906 w_k1_AXP_s117
                                 + 1.02747 w_k1_DIS_s117
                                 + 0.932516 w_k1_FDX_s117
                                 + 1.71068 w_k1_IBM_s117
                                 + 1.02442 w_k1_INTC_s117
                                 + 1.37344 w_k1_KO_s117 + 1.66959 w_k1_MAT_s117
                                 + 0.959659 w_k1_MSFT_s117
                                 + 0.909935 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.18197 w_k1_ADM_s117
                                 + 1.49247 w_k1_AXP_s117 + 1.134 w_k1_DIS_s117
                                 + 1.33972 w_k1_FDX_s117 + 1.0305 w_k1_IBM_s117
                                 + 1.29118 w_k1_INTC_s117 + 1.0696 w_k1_KO_s117
                                 + 1.09046 w_k1_MAT_s117
                                 + 1.26692 w_k1_MSFT_s117
                                 + 1.07139 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.87648 w_k1_ADM_s117
                                 + 0.891481 w_k1_AXP_s117
                                 + 0.674585 w_k1_DIS_s117
                                 + 0.890706 w_k1_FDX_s117
                                 + 0.669162 w_k1_IBM_s117
                                 + 0.88641 w_k1_INTC_s117
                                 + 0.696556 w_k1_KO_s117
                                 + 0.764833 w_k1_MAT_s117
                                 + 0.854381 w_k1_MSFT_s117
                                 + 0.976043 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.027836189223813 w_k1_ADM_s000
                                 + 0.00706909679637893 w_k1_ADM_s001
                                 + 0.00909626448649716 w_k1_ADM_s002
                                 + 0.0180154625561869 w_k1_ADM_s003
                                 + 0.0198321733308338 w_k1_ADM_s004
                                 + 0.0217933357098599 w_k1_ADM_s005
                                 + 0.0223765210200752 w_k1_ADM_s006
                                 + 0.00780655080702152 w_k1_ADM_s007
                                 + 0.0889269400740387 w_k1_ADM_s010
                                 + 0.0606307605775975 w_k1_ADM_s011
                                 + 0.0732725515133759 w_k1_ADM_s012
                                 + 0.0367615008498324 w_k1_ADM_s013
                                 + 0.0734766674552889 w_k1_ADM_s014
                                 + 0.0968479552719739 w_k1_ADM_s015
                                 + 0.0474087618213587 w_k1_ADM_s016
                                 + 0.0227891660395345 w_k1_ADM_s017
                                 + 0.0075328906324779 w_k1_ADM_s100
                                 + 0.0301862762833094 w_k1_ADM_s101
                                 + 0.0121662458356148 w_k1_ADM_s102
                                 + 0.0147398901676142 w_k1_ADM_s103
                                 + 0.0135310910901371 w_k1_ADM_s104
                                 + 0.00928481917418213 w_k1_ADM_s105
                                 + 0.0236718398437437 w_k1_ADM_s106
                                 + 0.0103764490238487 w_k1_ADM_s107
                                 + 0.0217137237026462 w_k1_ADM_s110
                                 + 0.0275803069978841 w_k1_ADM_s111
                                 + 0.0276437374121686 w_k1_ADM_s112
                                 + 0.0654846417608054 w_k1_ADM_s113
                                 + 0.0342655339785307 w_k1_ADM_s114
                                 + 0.0272988457207577 w_k1_ADM_s115
                                 + 0.0336933194473193 w_k1_ADM_s116
                                 + 0.0260122721759341 w_k1_ADM_s117
                                 + 0.0269129335953412 w_k1_AXP_s000
                                 + 0.00821412759980213 w_k1_AXP_s001
                                 + 0.0101145873032775 w_k1_AXP_s002
                                 + 0.0186553237377394 w_k1_AXP_s003
                                 + 0.0230310496361783 w_k1_AXP_s004
                                 + 0.0237413721835321 w_k1_AXP_s005
                                 + 0.0209641746282391 w_k1_AXP_s006
                                 + 0.00780755878621949 w_k1_AXP_s007
                                 + 0.0854042393828208 w_k1_AXP_s010
                                 + 0.0574082421475078 w_k1_AXP_s011
                                 + 0.074784400123962 w_k1_AXP_s012
                                 + 0.0443944372996144 w_k1_AXP_s013
                                 + 0.0746486009829946 w_k1_AXP_s014
                                 + 0.0936424456828739 w_k1_AXP_s015
                                 + 0.0461533967729939 w_k1_AXP_s016
                                 + 0.0227975240127448 w_k1_AXP_s017
                                 + 0.00767957324752762 w_k1_AXP_s100
                                 + 0.0284029645715574 w_k1_AXP_s101
                                 + 0.0120874734564319 w_k1_AXP_s102
                                 + 0.0170788364999327 w_k1_AXP_s103
                                 + 0.0152008114809409 w_k1_AXP_s104
                                 + 0.0102152514139628 w_k1_AXP_s105
                                 + 0.0239177647260225 w_k1_AXP_s106
                                 + 0.0143619690142805 w_k1_AXP_s107
                                 + 0.0254421899266554 w_k1_AXP_s110
                                 + 0.0273960702804579 w_k1_AXP_s111
                                 + 0.0291546579076774 w_k1_AXP_s112
                                 + 0.0567915834698571 w_k1_AXP_s113
                                 + 0.0404498856553213 w_k1_AXP_s114
                                 + 0.0281251093948064 w_k1_AXP_s115
                                 + 0.0370136526365801 w_k1_AXP_s116
                                 + 0.0292698419868773 w_k1_AXP_s117
                                 + 0.0273954667175316 w_k1_DIS_s000
                                 + 0.00760977698028314 w_k1_DIS_s001
                                 + 0.0107706237854554 w_k1_DIS_s002
                                 + 0.0193788385669526 w_k1_DIS_s003
                                 + 0.0216233072501582 w_k1_DIS_s004
                                 + 0.0216057802390338 w_k1_DIS_s005
                                 + 0.0204625467891426 w_k1_DIS_s006
                                 + 0.00890227625703289 w_k1_DIS_s007
                                 + 0.0876948020098456 w_k1_DIS_s010
                                 + 0.0651864880707626 w_k1_DIS_s011
                                 + 0.0732146523838393 w_k1_DIS_s012
                                 + 0.0440921486610273 w_k1_DIS_s013
                                 + 0.0719204099352332 w_k1_DIS_s014
                                 + 0.101007340393898 w_k1_DIS_s015
                                 + 0.0444028488300526 w_k1_DIS_s016
                                 + 0.0231741810277445 w_k1_DIS_s017
                                 + 0.00738300452309732 w_k1_DIS_s100
                                 + 0.0263220517096802 w_k1_DIS_s101
                                 + 0.0113743755364254 w_k1_DIS_s102
                                 + 0.0157888763364523 w_k1_DIS_s103
                                 + 0.0164305051175279 w_k1_DIS_s104
                                 + 0.00984560607150032 w_k1_DIS_s105
                                 + 0.0238952007878206 w_k1_DIS_s106
                                 + 0.01404069426385 w_k1_DIS_s107
                                 + 0.0240905087648632 w_k1_DIS_s110
                                 + 0.0280455731019423 w_k1_DIS_s111
                                 + 0.0296286032349465 w_k1_DIS_s112
                                 + 0.0642212863980844 w_k1_DIS_s113
                                 + 0.0397205047531594 w_k1_DIS_s114
                                 + 0.026676108590045 w_k1_DIS_s115
                                 + 0.0345631222982474 w_k1_DIS_s116
                                 + 0.0265335924116749 w_k1_DIS_s117
                                 + 0.0255370906572554 w_k1_FDX_s000
                                 + 0.00750529038877881 w_k1_FDX_s001
                                 + 0.0105283712858665 w_k1_FDX_s002
                                 + 0.0191459447251195 w_k1_FDX_s003
                                 + 0.0223203987208158 w_k1_FDX_s004
                                 + 0.0215886774929794 w_k1_FDX_s005
                                 + 0.0217428078085939 w_k1_FDX_s006
                                 + 0.00820371278596379 w_k1_FDX_s007
                                 + 0.0884358828233107 w_k1_FDX_s010
                                 + 0.0585194385172106 w_k1_FDX_s011
                                 + 0.0748624934588817 w_k1_FDX_s012
                                 + 0.0418525609177792 w_k1_FDX_s013
                                 + 0.0738265143475675 w_k1_FDX_s014
                                 + 0.0966566966370449 w_k1_FDX_s015
                                 + 0.041777678985682 w_k1_FDX_s016
                                 + 0.0223157480195952 w_k1_FDX_s017
                                 + 0.00744198887302539 w_k1_FDX_s100
                                 + 0.0272169463066226 w_k1_FDX_s101
                                 + 0.0117794639023384 w_k1_FDX_s102
                                 + 0.016135161785837 w_k1_FDX_s103
                                 + 0.0155191839709245 w_k1_FDX_s104
                                 + 0.00999209359019558 w_k1_FDX_s105
                                 + 0.0228998762355275 w_k1_FDX_s106
                                 + 0.0135981692357556 w_k1_FDX_s107
                                 + 0.0235356079895119 w_k1_FDX_s110
                                 + 0.0272027916152031 w_k1_FDX_s111
                                 + 0.0294685873597137 w_k1_FDX_s112
                                 + 0.0569316634596097 w_k1_FDX_s113
                                 + 0.0375965923712488 w_k1_FDX_s114
                                 + 0.0280326453057534 w_k1_FDX_s115
                                 + 0.034902702749595 w_k1_FDX_s116
                                 + 0.0280714765743151 w_k1_FDX_s117
                                 + 0.0262211492623664 w_k1_IBM_s000
                                 + 0.00805610387986471 w_k1_IBM_s001
                                 + 0.0108511529444711 w_k1_IBM_s002
                                 + 0.0183011320259063 w_k1_IBM_s003
                                 + 0.0212952253524151 w_k1_IBM_s004
                                 + 0.0215204688135706 w_k1_IBM_s005
                                 + 0.0213174173165375 w_k1_IBM_s006
                                 + 0.00852177663133263 w_k1_IBM_s007
                                 + 0.0855488480141754 w_k1_IBM_s010
                                 + 0.0590270821853393 w_k1_IBM_s011
                                 + 0.0721702160913672 w_k1_IBM_s012
                                 + 0.0406363676215277 w_k1_IBM_s013
                                 + 0.0743635471614594 w_k1_IBM_s014
                                 + 0.0993379154101576 w_k1_IBM_s015
                                 + 0.0402525315751253 w_k1_IBM_s016
                                 + 0.0225039507113205 w_k1_IBM_s017
                                 + 0.00749245791488073 w_k1_IBM_s100
                                 + 0.0265786690755952 w_k1_IBM_s101
                                 + 0.0115444485017742 w_k1_IBM_s102
                                 + 0.0164797085303546 w_k1_IBM_s103
                                 + 0.0156651265108021 w_k1_IBM_s104
                                 + 0.010437513059485 w_k1_IBM_s105
                                 + 0.0236528686693491 w_k1_IBM_s106
                                 + 0.0136265945715172 w_k1_IBM_s107
                                 + 0.0240627459093009 w_k1_IBM_s110
                                 + 0.027981003074802 w_k1_IBM_s111
                                 + 0.0282725726463197 w_k1_IBM_s112
                                 + 0.0575988288516817 w_k1_IBM_s113
                                 + 0.0386284635987245 w_k1_IBM_s114
                                 + 0.0274346016788987 w_k1_IBM_s115
                                 + 0.0342515151586671 w_k1_IBM_s116
                                 + 0.028162442744916 w_k1_IBM_s117
                                 + 0.0244539515870085 w_k1_INTC_s000
                                 + 0.00646851486091149 w_k1_INTC_s001
                                 + 0.0106897732906599 w_k1_INTC_s002
                                 + 0.0193178325712926 w_k1_INTC_s003
                                 + 0.0224299821946456 w_k1_INTC_s004
                                 + 0.0210274915988169 w_k1_INTC_s005
                                 + 0.0224646899520436 w_k1_INTC_s006
                                 + 0.00762123927619679 w_k1_INTC_s007
                                 + 0.0833201931543567 w_k1_INTC_s010
                                 + 0.0553287909758433 w_k1_INTC_s011
                                 + 0.0782777656858803 w_k1_INTC_s012
                                 + 0.0395620174710578 w_k1_INTC_s013
                                 + 0.0784246211750128 w_k1_INTC_s014
                                 + 0.0975194750894023 w_k1_INTC_s015
                                 + 0.0395752642961379 w_k1_INTC_s016
                                 + 0.0226378824600061 w_k1_INTC_s017
                                 + 0.00700456002553179 w_k1_INTC_s100
                                 + 0.0235006002284659 w_k1_INTC_s101
                                 + 0.00994349607764265 w_k1_INTC_s102
                                 + 0.0141938320771328 w_k1_INTC_s103
                                 + 0.015204564363707 w_k1_INTC_s104
                                 + 0.00747262683497258 w_k1_INTC_s105
                                 + 0.0233741142367303 w_k1_INTC_s106
                                 + 0.0123811258220417 w_k1_INTC_s107
                                 + 0.0261356691748591 w_k1_INTC_s110
                                 + 0.0274817526812606 w_k1_INTC_s111
                                 + 0.0272998306309494 w_k1_INTC_s112
                                 + 0.0497350099893973 w_k1_INTC_s113
                                 + 0.0405719248618814 w_k1_INTC_s114
                                 + 0.0291844780921063 w_k1_INTC_s115
                                 + 0.0337145553757892 w_k1_INTC_s116
                                 + 0.0293188243020538 w_k1_INTC_s117
                                 + 0.0276635613036817 w_k1_KO_s000
                                 + 0.00708265428634988 w_k1_KO_s001
                                 + 0.0107783836280038 w_k1_KO_s002
                                 + 0.0180666549989042 w_k1_KO_s003
                                 + 0.0227138257509489 w_k1_KO_s004
                                 + 0.0192752597406775 w_k1_KO_s005
                                 + 0.0218664685273461 w_k1_KO_s006
                                 + 0.00908105670954069 w_k1_KO_s007
                                 + 0.0828025046849154 w_k1_KO_s010
                                 + 0.0628529466474558 w_k1_KO_s011
                                 + 0.0733825101057932 w_k1_KO_s012
                                 + 0.0397279153366526 w_k1_KO_s013
                                 + 0.0800135534688519 w_k1_KO_s014
                                 + 0.0969420760135821 w_k1_KO_s015
                                 + 0.0370902404271342 w_k1_KO_s016
                                 + 0.0219148508846753 w_k1_KO_s017
                                 + 0.00707923435904468 w_k1_KO_s100
                                 + 0.0272473282383414 w_k1_KO_s101
                                 + 0.0101711607885796 w_k1_KO_s102
                                 + 0.0154113207920968 w_k1_KO_s103
                                 + 0.0153581540768797 w_k1_KO_s104
                                 + 0.0095278203083381 w_k1_KO_s105
                                 + 0.0232330369725938 w_k1_KO_s106
                                 + 0.0146601182618661 w_k1_KO_s107
                                 + 0.023818849374579 w_k1_KO_s110
                                 + 0.0323026503027977 w_k1_KO_s111
                                 + 0.0286833445383401 w_k1_KO_s112
                                 + 0.0565339938089233 w_k1_KO_s113
                                 + 0.0377996157129516 w_k1_KO_s114
                                 + 0.0286080541132047 w_k1_KO_s115
                                 + 0.0303786890533037 w_k1_KO_s116
                                 + 0.0282898071238342 w_k1_KO_s117
                                 + 0.0339306550417263 w_k1_MAT_s000
                                 + 0.0115411600118372 w_k1_MAT_s001
                                 + 0.0114247058882758 w_k1_MAT_s002
                                 + 0.0147886601855058 w_k1_MAT_s003
                                 + 0.0216271829815344 w_k1_MAT_s004
                                 + 0.0223350929326582 w_k1_MAT_s005
                                 + 0.0227306092361788 w_k1_MAT_s006
                                 + 0.00836843161258045 w_k1_MAT_s007
                                 + 0.0860682836944658 w_k1_MAT_s010
                                 + 0.0641879278648042 w_k1_MAT_s011
                                 + 0.0724577813350849 w_k1_MAT_s012
                                 + 0.035786723254031 w_k1_MAT_s013
                                 + 0.0700780434630498 w_k1_MAT_s014
                                 + 0.0989039974449027 w_k1_MAT_s015
                                 + 0.0382685359146588 w_k1_MAT_s016
                                 + 0.0226593411769992 w_k1_MAT_s017
                                 + 0.0075835909926224 w_k1_MAT_s100
                                 + 0.0362161620178083 w_k1_MAT_s101
                                 + 0.0114671196525245 w_k1_MAT_s102
                                 + 0.0190644939077311 w_k1_MAT_s103
                                 + 0.0147023503905885 w_k1_MAT_s104
                                 + 0.0115043530141292 w_k1_MAT_s105
                                 + 0.0213799716748635 w_k1_MAT_s106
                                 + 0.0150430207539206 w_k1_MAT_s107
                                 + 0.0182266499678638 w_k1_MAT_s110
                                 + 0.0330266565714161 w_k1_MAT_s111
                                 + 0.0312347166649742 w_k1_MAT_s112
                                 + 0.0620174322521636 w_k1_MAT_s113
                                 + 0.0406230376921332 w_k1_MAT_s114
                                 + 0.0262533707531072 w_k1_MAT_s115
                                 + 0.0321666638373473 w_k1_MAT_s116
                                 + 0.0281683111292039 w_k1_MAT_s117
                                 + 0.0265756456002224 w_k1_MSFT_s000
                                 + 0.0076524811975547 w_k1_MSFT_s001
                                 + 0.0115234364991628 w_k1_MSFT_s002
                                 + 0.0200667227261049 w_k1_MSFT_s003
                                 + 0.0229922804488136 w_k1_MSFT_s004
                                 + 0.0212792186284348 w_k1_MSFT_s005
                                 + 0.0207276114491472 w_k1_MSFT_s006
                                 + 0.00915786871539281 w_k1_MSFT_s007
                                 + 0.0844728793906375 w_k1_MSFT_s010
                                 + 0.0553550317398698 w_k1_MSFT_s011
                                 + 0.0726670317603811 w_k1_MSFT_s012
                                 + 0.0431988056761244 w_k1_MSFT_s013
                                 + 0.0838328570613494 w_k1_MSFT_s014
                                 + 0.0990099351417048 w_k1_MSFT_s015
                                 + 0.0371548006165763 w_k1_MSFT_s016
                                 + 0.0213852468204532 w_k1_MSFT_s017
                                 + 0.00673686475368628 w_k1_MSFT_s100
                                 + 0.0310302430958232 w_k1_MSFT_s101
                                 + 0.00995042363373218 w_k1_MSFT_s102
                                 + 0.0168082528906345 w_k1_MSFT_s103
                                 + 0.0146837288687489 w_k1_MSFT_s104
                                 + 0.00906860163967334 w_k1_MSFT_s105
                                 + 0.0238829200090308 w_k1_MSFT_s106
                                 + 0.011799583288444 w_k1_MSFT_s107
                                 + 0.0243943867439756 w_k1_MSFT_s110
                                 + 0.025715386802327 w_k1_MSFT_s111
                                 + 0.026165921138412 w_k1_MSFT_s112
                                 + 0.057497068443907 w_k1_MSFT_s113
                                 + 0.0358037505773367 w_k1_MSFT_s114
                                 + 0.0241275045028133 w_k1_MSFT_s115
                                 + 0.03101764257027 w_k1_MSFT_s116
                                 + 0.0268461843147976 w_k1_MSFT_s117
                                 + 0.0272039821955176 w_k1_PG_s000
                                 + 0.00701791381662389 w_k1_PG_s001
                                 + 0.0093407405429724 w_k1_PG_s002
                                 + 0.0171108775121046 w_k1_PG_s003
                                 + 0.021476105814998 w_k1_PG_s004
                                 + 0.020159380711404 w_k1_PG_s005
                                 + 0.0211648417985656 w_k1_PG_s006
                                 + 0.00896988165833616 w_k1_PG_s007
                                 + 0.0883781823722942 w_k1_PG_s010
                                 + 0.0590823272269182 w_k1_PG_s011
                                 + 0.0760651427773824 w_k1_PG_s012
                                 + 0.0402289282818673 w_k1_PG_s013
                                 + 0.0767874956458153 w_k1_PG_s014
                                 + 0.096592104808211 w_k1_PG_s015
                                 + 0.0417357542091557 w_k1_PG_s016
                                 + 0.0230706681131708 w_k1_PG_s017
                                 + 0.00704860595319099 w_k1_PG_s100
                                 + 0.0270015182067038 w_k1_PG_s101
                                 + 0.0101949415870363 w_k1_PG_s102
                                 + 0.0161990884798394 w_k1_PG_s103
                                 + 0.0161583460412619 w_k1_PG_s104
                                 + 0.00894221512777602 w_k1_PG_s105
                                 + 0.0219432921979654 w_k1_PG_s106
                                 + 0.0121686523621292 w_k1_PG_s107
                                 + 0.0247788334819174 w_k1_PG_s110
                                 + 0.0272867543587202 w_k1_PG_s111
                                 + 0.0299471349173287 w_k1_PG_s112
                                 + 0.0604599473811372 w_k1_PG_s113
                                 + 0.0362809354414846 w_k1_PG_s114
                                 + 0.0266158926624736 w_k1_PG_s115
                                 + 0.033724237883299 w_k1_PG_s116
                                 + 0.0266068521317663 w_k1_PG_s117 >= 1.19
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
