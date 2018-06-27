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
 w_k1_ADM_s0_return_constraint:  1.05662 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.982943 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.23913 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.959935 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.28097 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.900974 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.776943 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.901795 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.25441 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.865096 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.963797 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.20346 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.12777 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.996455 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.952571 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.873711 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.818415 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.951223 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.12405 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.04251 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.660882 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.890723 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.997086 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.915441 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.02465 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.887239 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.823508 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.804819 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.920923 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.34114 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.0048 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.08443 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.25321 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.27248 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.07452 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.777964 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.924517 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.33966 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.18411 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.918209 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.34242 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.877317 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.35233 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.953143 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.832036 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.807081 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.16902 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.638929 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.782251 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.30343 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.08614 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.04341 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.04278 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.919181 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.645722 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.39877 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.03278 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.61204 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.733477 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.797033 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.3779 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.985074 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.10543 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.104 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.861228 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.864874 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.06511 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.70077 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.937384 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.816471 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.46605 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.18585 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.04182 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.582076 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.953456 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.45678 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.25962 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.921608 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.33644 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.800956 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.44623 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.820279 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.757099 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.863831 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.23094 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.599623 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.03849 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.41701 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.52594 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.914575 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.13632 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.915101 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.630281 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.42077 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.07796 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.72014 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.637471 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.772972 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.52697 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.948057 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.41628 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.924903 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.00618 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.719093 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.12861 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.63493 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.1377 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.858007 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.5103 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.48283 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.3725 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.584231 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.892395 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.887 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.08879 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.966859 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.29759 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.888642 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.26651 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.89581 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.809908 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.940201 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.00803 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.743031 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.956383 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.26912 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.23739 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.0877 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.04873 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.849318 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.709803 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.25201 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.21069 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.30894 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.693282 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.921986 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.19554 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.83879 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.22485 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.10251 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.01137 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.784554 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.994822 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.51732 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.03771 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.01226 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.2315 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.38735 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.06011 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.682221 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.905841 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.51699 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.2813 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.867519 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.99307 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.01925 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.29635 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.936911 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.896935 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.608878 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.54328 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.568337 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.882154 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.17629 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.81197 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.828556 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.900552 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.20853 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.617573 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.1205 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.681216 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.76413 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.886457 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.614159 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.5677 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.6263 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.777733 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.768087 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.668696 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.939779 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.14446 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.03236 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.840634 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.553804 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.59392 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.749072 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.49861 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.907144 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.09148 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.14262 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10876 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.01314 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.20825 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.90921 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.19843 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.888278 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.883487 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.932943 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.870656 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.73208 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.904314 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.2803 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.06156 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.976468 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.08223 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.839263 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.710706 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.11182 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.22771 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.35129 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.81893 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.971388 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.00283 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.831666 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.15767 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.999378 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.00412 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.708655 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.02653 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.27933 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.06458 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.908998 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.03911 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.24021 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.14399 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.720029 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.01125 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.22601 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.14369 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.956936 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.18141 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.00898 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.14815 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.863588 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.86381 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.848622 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.24877 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.630156 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.03711 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.24412 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.0482 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.872241 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.940654 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.01321 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.705229 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.16256 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.858001 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.53715 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.743696 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.723566 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.55646 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.30865 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.02531 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.899134 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.979208 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.931169 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.15426 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.25641 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.944524 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.694947 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.32132 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.15684 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.25119 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.680099 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.05412 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.23425 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.29136 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.900389 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.05402 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.10353 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.37466 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.809716 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.02099 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.677582 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.60494 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.606389 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.906851 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.53386 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.909119 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.07848 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.75285 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.985369 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.556305 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.06684 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.559229 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.54972 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.734802 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.475842 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.53181 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.58339 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.90828 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.805418 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.711632 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.760903 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.18521 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.04655 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.936691 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.599594 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.87798 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.05976 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.41235 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.711644 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.32449 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.34047 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.08913 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.00803 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.25742 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.912242 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.18536 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.873457 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.842928 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.903192 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.02105 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.781845 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.96816 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.23178 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.12082 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.03989 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.02444 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.870582 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.750392 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.21698 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.21478 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.31946 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.753542 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.895197 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.1742 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.862616 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.24693 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.974274 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.999232 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.74009 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.00488 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.35309 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.07264 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.941925 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.1472 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.35626 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.18411 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.699635 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.840696 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.47168 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.10849 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.959257 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.04654 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.982062 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.11815 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.961398 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.914447 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.908787 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.11662 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.866193 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.05504 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.23061 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.15467 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.15824 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.00877 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.900395 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.687156 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.09093 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.953657 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.2302 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.82491 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.765174 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.24385 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.07669 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.14101 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.98129 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.984239 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.983553 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.13261 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.22758 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.981091 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.846641 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.19839 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.05433 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.996664 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.755482 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.913044 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.14196 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.290000449222826 z_k1_s0000
                                 - 0.200000409036677 z_k1_s0001
                                 - 0.22900045395896 z_k1_s0002
                                 - 0.315999629523435 z_k1_s0003
                                 - 0.671002825509646 z_k1_s0004
                                 - 0.143000287665212 z_k1_s0005
                                 - 0.351000924087879 z_k1_s0006
                                 - 0.198000472090476 z_k1_s0007
                                 - 0.0570000800590163 z_k1_s0010
                                 - 0.0950002777651453 z_k1_s0011
                                 - 0.0540000804032823 z_k1_s0012
                                 - 0.113000102499407 z_k1_s0013
                                 - 0.102000161495097 z_k1_s0014
                                 - 0.0850000479791763 z_k1_s0015
                                 - 0.137000446145563 z_k1_s0016
                                 - 0.223000378543222 z_k1_s0017
                                 - 0.310999934684996 z_k1_s0020
                                 - 0.305999909668195 z_k1_s0021
                                 - 0.631999919403596 z_k1_s0022
                                 - 0.234999928589626 z_k1_s0023
                                 - 0.395000729127281 z_k1_s0024
                                 - 0.4009997613874 z_k1_s0025
                                 - 0.647000618054028 z_k1_s0026
                                 - 0.190999957881782 z_k1_s0027
                                 - 0.121000165137189 z_k1_s0030
                                 - 0.187000914503815 z_k1_s0031
                                 - 0.273000035166922 z_k1_s0032
                                 - 0.136999614255386 z_k1_s0033
                                 - 0.154000674120682 z_k1_s0034
                                 - 0.132000155731446 z_k1_s0035
                                 - 0.266000309752756 z_k1_s0036
                                 - 0.0730000718802501 z_k1_s0037
                                 - 0.0519999935701915 z_k1_s0040
                                 - 0.040999985653419 z_k1_s0041
                                 - 0.139999735304328 z_k1_s0042
                                 - 0.133000110339161 z_k1_s0043
                                 - 0.0859998718583775 z_k1_s0044
                                 - 0.137000010890701 z_k1_s0045
                                 - 0.115000155857178 z_k1_s0046
                                 - 0.0999999257888214 z_k1_s0047
                                 - 0.223000027914339 z_k1_s0050
                                 - 0.388000184625907 z_k1_s0051
                                 - 0.229000078645689 z_k1_s0052
                                 - 0.429000531290134 z_k1_s0053
                                 - 0.236000137832264 z_k1_s0054
                                 - 0.702999752020386 z_k1_s0055
                                 - 0.447000683484184 z_k1_s0056
                                 - 0.441000632752834 z_k1_s0057
                                 - 0.0699998514671049 z_k1_s0060
                                 - 0.0859999738595312 z_k1_s0061
                                 - 0.119000226293911 z_k1_s0062
                                 - 0.110000157448171 z_k1_s0063
                                 - 0.0550000103241093 z_k1_s0064
                                 - 0.0329999788344752 z_k1_s0065
                                 - 0.066999965318477 z_k1_s0066
                                 - 0.143999733532316 z_k1_s0067
                                 - 0.17999979985234 z_k1_s0070
                                 - 0.0700000458093255 z_k1_s0071
                                 - 0.0920003732581792 z_k1_s0072
                                 - 0.107999879911404 z_k1_s0073
                                 - 0.138000131886984 z_k1_s0074
                                 - 0.0830000408653342 z_k1_s0075
                                 - 0.0970002725873475 z_k1_s0076
                                 - 0.0880002825947305 z_k1_s0077
                                 - 0.896996513168175 z_k1_s0100
                                 - 0.944998305398178 z_k1_s0101
                                 - 0.304998869652099 z_k1_s0102
                                 - 0.395997496953626 z_k1_s0103
                                 - 0.197999516874319 z_k1_s0104
                                 - 0.397000255699586 z_k1_s0105
                                 - 0.270999200983985 z_k1_s0106
                                 - 0.432999678878322 z_k1_s0107
                                 - 0.605997943621116 z_k1_s0110
                                 - 0.825997917595196 z_k1_s0111
                                 - 0.618000115908853 z_k1_s0112
                                 - 1.04899713514226 z_k1_s0113
                                 - 0.342999628641674 z_k1_s0114
                                 - 1.06999703814895 z_k1_s0115
                                 - 0.429999523840591 z_k1_s0116
                                 - 0.255999733442757 z_k1_s0117
                                 - 0.142999938417857 z_k1_s0120
                                 - 0.34699982473706 z_k1_s0121
                                 - 0.354999378114649 z_k1_s0122
                                 - 0.208000051807675 z_k1_s0123
                                 - 0.123999881347007 z_k1_s0124
                                 - 0.156999837228076 z_k1_s0125
                                 - 0.403999801549771 z_k1_s0126
                                 - 0.20599967746368 z_k1_s0127
                                 - 0.934999033725433 z_k1_s0130
                                 - 0.906998470085979 z_k1_s0131
                                 - 0.391999648758643 z_k1_s0132
                                 - 0.224999982956223 z_k1_s0133
                                 - 0.435999684092719 z_k1_s0134
                                 - 0.592998363624504 z_k1_s0135
                                 - 0.684999561440352 z_k1_s0136
                                 - 0.405999472678719 z_k1_s0137
                                 - 0.20700092151252 z_k1_s0140
                                 - 0.340000029594773 z_k1_s0141
                                 - 0.242999173774705 z_k1_s0142
                                 - 0.349000069660535 z_k1_s0143
                                 - 0.113000015447677 z_k1_s0144
                                 - 0.266000124165368 z_k1_s0145
                                 - 0.262000954358746 z_k1_s0146
                                 - 0.128000145823976 z_k1_s0147
                                 - 0.325000086385176 z_k1_s0150
                                 - 0.243000189357574 z_k1_s0151
                                 - 0.577999654560543 z_k1_s0152
                                 - 0.558001676725225 z_k1_s0153
                                 - 0.251000160291968 z_k1_s0154
                                 - 0.331999963477737 z_k1_s0155
                                 - 0.892999972227793 z_k1_s0156
                                 - 0.718999239754264 z_k1_s0157
                                 - 0.31699977453032 z_k1_s0160
                                 - 0.102999979724358 z_k1_s0161
                                 - 0.506999569264586 z_k1_s0162
                                 - 0.387999918825738 z_k1_s0163
                                 - 0.255000309511732 z_k1_s0164
                                 - 0.513999548702166 z_k1_s0165
                                 - 0.20299989768976 z_k1_s0166
                                 - 0.183000062296667 z_k1_s0167
                                 - 0.722997652060307 z_k1_s0170
                                 - 0.363997609325445 z_k1_s0171
                                 - 0.335998972451092 z_k1_s0172
                                 - 0.179999471427312 z_k1_s0173
                                 - 0.283999138776499 z_k1_s0174
                                 - 0.387998192927331 z_k1_s0175
                                 - 0.204999568347315 z_k1_s0176
                                 - 0.585999941651122 z_k1_s0177
                                 - 0.219999437525417 z_k1_s1000
                                 - 0.113999892081165 z_k1_s1001
                                 - 0.247000169975445 z_k1_s1002
                                 - 0.147999278990175 z_k1_s1003
                                 - 0.0709999469557988 z_k1_s1004
                                 - 0.0729999663150958 z_k1_s1005
                                 - 0.173999934460624 z_k1_s1006
                                 - 0.299000134917717 z_k1_s1007
                                 - 0.216000222560949 z_k1_s1010
                                 - 0.094999945297757 z_k1_s1011
                                 - 0.148000125958448 z_k1_s1012
                                 - 0.23099942200331 z_k1_s1013
                                 - 0.23500035492039 z_k1_s1014
                                 - 0.209000022954955 z_k1_s1015
                                 - 0.127999903669627 z_k1_s1016
                                 - 0.170999758236073 z_k1_s1017
                                 - 0.130999272549797 z_k1_s1020
                                 - 0.270999628158387 z_k1_s1021
                                 - 0.0979998955265993 z_k1_s1022
                                 - 0.0869998675859244 z_k1_s1023
                                 - 0.16899913347252 z_k1_s1024
                                 - 0.222999741835051 z_k1_s1025
                                 - 0.0929998828262925 z_k1_s1026
                                 - 0.105999798047286 z_k1_s1027
                                 - 0.370998819582179 z_k1_s1030
                                 - 0.292997790360404 z_k1_s1031
                                 - 0.363998311064148 z_k1_s1032
                                 - 0.244998322218686 z_k1_s1033
                                 - 0.248999540224819 z_k1_s1034
                                 - 0.250999067232751 z_k1_s1035
                                 - 0.258999339254749 z_k1_s1036
                                 - 0.131999511185344 z_k1_s1037
                                 - 0.188998961313565 z_k1_s1040
                                 - 0.142999165607894 z_k1_s1041
                                 - 0.228998144235185 z_k1_s1042
                                 - 0.127999257138512 z_k1_s1043
                                 - 0.287997707815192 z_k1_s1044
                                 - 0.175999002440318 z_k1_s1045
                                 - 0.229998978528352 z_k1_s1046
                                 - 0.526996555025608 z_k1_s1047
                                 - 1.14899529114564 z_k1_s1050
                                 - 0.658001561729169 z_k1_s1051
                                 - 0.481998899088117 z_k1_s1052
                                 - 0.333999393855377 z_k1_s1053
                                 - 0.699997581357696 z_k1_s1054
                                 - 0.532999046627867 z_k1_s1055
                                 - 0.4729988388694 z_k1_s1056
                                 - 1.48199984363386 z_k1_s1057
                                 - 0.155000323518141 z_k1_s1060
                                 - 0.311001016938241 z_k1_s1061
                                 - 0.244999318843013 z_k1_s1062
                                 - 0.213000398599699 z_k1_s1063
                                 - 0.213000398599699 z_k1_s1064
                                 - 0.376000563528835 z_k1_s1065
                                 - 0.33900112997773 z_k1_s1066
                                 - 0.375000639027545 z_k1_s1067
                                 - 0.294998665364542 z_k1_s1070
                                 - 0.353000124423999 z_k1_s1071
                                 - 0.219999674508354 z_k1_s1072
                                 - 0.303999334666345 z_k1_s1073
                                 - 0.276999302758587 z_k1_s1074
                                 - 0.173999832383773 z_k1_s1075
                                 - 0.106999877619929 z_k1_s1076
                                 - 0.245999851604539 z_k1_s1077
                                 - 0.487001132154876 z_k1_s1100
                                 - 0.581001138913288 z_k1_s1101
                                 - 0.627000933241444 z_k1_s1102
                                 - 1.68900276850499 z_k1_s1103
                                 - 0.692001425532836 z_k1_s1104
                                 - 0.785001223240198 z_k1_s1105
                                 - 1.39100267741966 z_k1_s1106
                                 - 1.93300630386382 z_k1_s1107
                                 - 0.48699850263012 z_k1_s1110
                                 - 0.473001994507231 z_k1_s1111
                                 - 0.435999226032633 z_k1_s1112
                                 - 0.527000522845337 z_k1_s1113
                                 - 0.278999994517956 z_k1_s1114
                                 - 0.286999998660767 z_k1_s1115
                                 - 0.582998552343855 z_k1_s1116
                                 - 0.9270007299859 z_k1_s1117
                                 - 0.265999839811116 z_k1_s1120
                                 - 0.347001035614252 z_k1_s1121
                                 - 0.293999956212259 z_k1_s1122
                                 - 0.245000069010195 z_k1_s1123
                                 - 0.395998390816823 z_k1_s1124
                                 - 0.245999892310272 z_k1_s1125
                                 - 1.00500123904163 z_k1_s1126
                                 - 0.366000526815145 z_k1_s1127
                                 - 0.276000193208753 z_k1_s1130
                                 - 0.275000704219373 z_k1_s1131
                                 - 0.346999219489701 z_k1_s1132
                                 - 0.468999344255928 z_k1_s1133
                                 - 0.192000205258333 z_k1_s1134
                                 - 0.576001702176074 z_k1_s1135
                                 - 0.297999814985872 z_k1_s1136
                                 - 0.282999332137103 z_k1_s1137
                                 - 0.673999151326926 z_k1_s1140
                                 - 0.390998753413276 z_k1_s1141
                                 - 0.394998415616813 z_k1_s1142
                                 - 0.221999942306617 z_k1_s1143
                                 - 0.485002029202581 z_k1_s1144
                                 - 0.771000304767854 z_k1_s1145
                                 - 0.495999231261279 z_k1_s1146
                                 - 0.304000119683045 z_k1_s1147
                                 - 0.417000378203772 z_k1_s1150
                                 - 1.61400379830022 z_k1_s1151
                                 - 1.01499934207629 z_k1_s1152
                                 - 0.751998692283828 z_k1_s1153
                                 - 0.375000258200388 z_k1_s1154
                                 - 1.25300144529766 z_k1_s1155
                                 - 0.715998792738384 z_k1_s1156
                                 - 0.974004002823216 z_k1_s1157
                                 - 0.638999413541287 z_k1_s1160
                                 - 2.01300247206732 z_k1_s1161
                                 - 0.904999645359173 z_k1_s1162
                                 - 0.813999641795121 z_k1_s1163
                                 - 0.503999429339661 z_k1_s1164
                                 - 1.41000068779433 z_k1_s1165
                                 - 2.04599622144864 z_k1_s1166
                                 - 1.26300149383661 z_k1_s1167
                                 - 0.711001570589331 z_k1_s1170
                                 - 0.223000191537006 z_k1_s1171
                                 - 0.470999731858086 z_k1_s1172
                                 - 0.284999992342233 z_k1_s1173
                                 - 0.940000459693929 z_k1_s1174
                                 - 0.21000026048915 z_k1_s1175
                                 - 0.33700005363383 z_k1_s1176
                                 - 0.194000137907067 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.46961 w_k1_ADM_s000
                                 + 1.07056 w_k1_AXP_s000
                                 + 1.05986 w_k1_DIS_s000
                                 + 1.00144 w_k1_FDX_s000
                                 + 1.57504 w_k1_IBM_s000
                                 + 1.19597 w_k1_INTC_s000
                                 + 1.33209 w_k1_KO_s000 + 1.20937 w_k1_MAT_s000
                                 + 1.14852 w_k1_MSFT_s000
                                 + 1.08336 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.715177 w_k1_ADM_s000
                                 + 0.715774 w_k1_AXP_s000
                                 + 0.693768 w_k1_DIS_s000
                                 + 0.763425 w_k1_FDX_s000
                                 + 0.994308 w_k1_IBM_s000
                                 + 0.901633 w_k1_INTC_s000
                                 + 0.862932 w_k1_KO_s000
                                 + 0.874477 w_k1_MAT_s000
                                 + 0.839179 w_k1_MSFT_s000
                                 + 0.852986 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.20143 w_k1_ADM_s000
                                 + 0.819506 w_k1_AXP_s000
                                 + 0.834255 w_k1_DIS_s000
                                 + 0.943016 w_k1_FDX_s000
                                 + 0.787096 w_k1_IBM_s000
                                 + 0.94108 w_k1_INTC_s000
                                 + 0.865537 w_k1_KO_s000
                                 + 0.486643 w_k1_MAT_s000
                                 + 0.984597 w_k1_MSFT_s000
                                 + 1.0632 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.07369 w_k1_ADM_s000
                                 + 1.09372 w_k1_AXP_s000
                                 + 0.845609 w_k1_DIS_s000
                                 + 0.998696 w_k1_FDX_s000
                                 + 1.12694 w_k1_IBM_s000
                                 + 1.04882 w_k1_INTC_s000
                                 + 1.00697 w_k1_KO_s000 + 0.95019 w_k1_MAT_s000
                                 + 0.852508 w_k1_MSFT_s000
                                 + 1.00592 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.14226 w_k1_ADM_s000
                                 + 1.12562 w_k1_AXP_s000
                                 + 1.15698 w_k1_DIS_s000 + 1.2364 w_k1_FDX_s000
                                 + 1.01297 w_k1_IBM_s000
                                 + 1.00672 w_k1_INTC_s000
                                 + 1.01582 w_k1_KO_s000 + 0.99778 w_k1_MAT_s000
                                 + 1.12111 w_k1_MSFT_s000
                                 + 1.22879 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.20615 w_k1_ADM_s000
                                 + 0.742011 w_k1_AXP_s000
                                 + 0.74578 w_k1_DIS_s000
                                 + 0.96395 w_k1_FDX_s000
                                 + 0.649109 w_k1_IBM_s000
                                 + 0.896005 w_k1_INTC_s000
                                 + 0.787104 w_k1_KO_s000
                                 + 0.508555 w_k1_MAT_s000
                                 + 0.910442 w_k1_MSFT_s000
                                 + 0.90432 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.16174 w_k1_ADM_s000
                                 + 0.888867 w_k1_AXP_s000
                                 + 1.04739 w_k1_DIS_s000
                                 + 0.953494 w_k1_FDX_s000
                                 + 1.11148 w_k1_IBM_s000
                                 + 0.895952 w_k1_INTC_s000
                                 + 1.07502 w_k1_KO_s000
                                 + 0.881564 w_k1_MAT_s000
                                 + 0.951696 w_k1_MSFT_s000
                                 + 1.17215 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.701138 w_k1_ADM_s000
                                 + 0.609739 w_k1_AXP_s000
                                 + 0.54501 w_k1_DIS_s000
                                 + 0.761578 w_k1_FDX_s000
                                 + 0.699572 w_k1_IBM_s000
                                 + 0.605822 w_k1_INTC_s000
                                 + 0.658505 w_k1_KO_s000
                                 + 0.591008 w_k1_MAT_s000
                                 + 0.699522 w_k1_MSFT_s000
                                 + 0.899383 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.23461 w_k1_ADM_s001
                                 + 1.47956 w_k1_AXP_s001
                                 + 1.26098 w_k1_DIS_s001
                                 + 1.21956 w_k1_FDX_s001
                                 + 1.45129 w_k1_IBM_s001
                                 + 1.08787 w_k1_INTC_s001
                                 + 1.17888 w_k1_KO_s001 + 2.07443 w_k1_MAT_s001
                                 + 1.2143 w_k1_MSFT_s001 + 1.26718 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.927077 w_k1_ADM_s001
                                 + 1.04616 w_k1_AXP_s001
                                 + 0.918105 w_k1_DIS_s001
                                 + 0.863643 w_k1_FDX_s001
                                 + 1.26091 w_k1_IBM_s001
                                 + 0.9129 w_k1_INTC_s001
                                 + 0.979553 w_k1_KO_s001
                                 + 1.03032 w_k1_MAT_s001
                                 + 0.941302 w_k1_MSFT_s001
                                 + 1.00013 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.38167 w_k1_ADM_s001
                                 + 1.62017 w_k1_AXP_s001
                                 + 1.71418 w_k1_DIS_s001
                                 + 1.40226 w_k1_FDX_s001 + 1.4381 w_k1_IBM_s001
                                 + 1.04458 w_k1_INTC_s001
                                 + 1.28315 w_k1_KO_s001 + 1.56175 w_k1_MAT_s001
                                 + 1.18526 w_k1_MSFT_s001
                                 + 1.38387 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.17659 w_k1_ADM_s001
                                 + 0.937167 w_k1_AXP_s001
                                 + 0.933266 w_k1_DIS_s001
                                 + 0.999132 w_k1_FDX_s001
                                 + 1.12234 w_k1_IBM_s001
                                 + 1.11174 w_k1_INTC_s001
                                 + 0.88978 w_k1_KO_s001
                                 + 0.880141 w_k1_MAT_s001
                                 + 0.949014 w_k1_MSFT_s001
                                 + 1.04098 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.07332 w_k1_ADM_s001
                                 + 0.695344 w_k1_AXP_s001
                                 + 0.741017 w_k1_DIS_s001
                                 + 0.860097 w_k1_FDX_s001
                                 + 0.743029 w_k1_IBM_s001
                                 + 0.692153 w_k1_INTC_s001
                                 + 0.804059 w_k1_KO_s001
                                 + 0.726132 w_k1_MAT_s001
                                 + 0.857911 w_k1_MSFT_s001
                                 + 1.04945 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.747695 w_k1_ADM_s001
                                 + 0.78053 w_k1_AXP_s001
                                 + 0.632594 w_k1_DIS_s001
                                 + 0.748681 w_k1_FDX_s001
                                 + 0.808652 w_k1_IBM_s001
                                 + 0.775784 w_k1_INTC_s001
                                 + 0.646076 w_k1_KO_s001
                                 + 0.598561 w_k1_MAT_s001
                                 + 0.693892 w_k1_MSFT_s001
                                 + 0.759052 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.763991 w_k1_ADM_s001
                                 + 0.862052 w_k1_AXP_s001
                                 + 0.968567 w_k1_DIS_s001
                                 + 0.945146 w_k1_FDX_s001
                                 + 0.87127 w_k1_IBM_s001
                                 + 0.893639 w_k1_INTC_s001
                                 + 0.840623 w_k1_KO_s001
                                 + 0.888818 w_k1_MAT_s001
                                 + 1.07594 w_k1_MSFT_s001
                                 + 1.01113 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.37361 w_k1_ADM_s001
                                 + 1.2171 w_k1_AXP_s001 + 1.15647 w_k1_DIS_s001
                                 + 1.19121 w_k1_FDX_s001
                                 + 1.01237 w_k1_IBM_s001
                                 + 1.13854 w_k1_INTC_s001
                                 + 1.01085 w_k1_KO_s001
                                 + 0.757239 w_k1_MAT_s001
                                 + 1.17187 w_k1_MSFT_s001
                                 + 1.03119 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.783975 w_k1_ADM_s002
                                 + 0.982063 w_k1_AXP_s002
                                 + 0.966595 w_k1_DIS_s002
                                 + 1.05585 w_k1_FDX_s002
                                 + 0.718934 w_k1_IBM_s002
                                 + 1.21669 w_k1_INTC_s002
                                 + 0.930494 w_k1_KO_s002
                                 + 0.799052 w_k1_MAT_s002
                                 + 1.07778 w_k1_MSFT_s002
                                 + 0.940587 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.867785 w_k1_ADM_s002
                                 + 1.00372 w_k1_AXP_s002
                                 + 1.03349 w_k1_DIS_s002
                                 + 0.977663 w_k1_FDX_s002
                                 + 1.55588 w_k1_IBM_s002
                                 + 1.17875 w_k1_INTC_s002
                                 + 1.26738 w_k1_KO_s002 + 1.43877 w_k1_MAT_s002
                                 + 1.07166 w_k1_MSFT_s002
                                 + 1.14757 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.753584 w_k1_ADM_s002
                                 + 1.18038 w_k1_AXP_s002
                                 + 1.06146 w_k1_DIS_s002 + 1.0533 w_k1_FDX_s002
                                 + 1.2125 w_k1_IBM_s002
                                 + 1.26213 w_k1_INTC_s002
                                 + 1.11252 w_k1_KO_s002 + 1.16354 w_k1_MAT_s002
                                 + 1.05651 w_k1_MSFT_s002
                                 + 0.986851 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.691091 w_k1_ADM_s002
                                 + 0.6474 w_k1_AXP_s002
                                 + 0.573741 w_k1_DIS_s002
                                 + 0.657789 w_k1_FDX_s002
                                 + 0.761026 w_k1_IBM_s002
                                 + 1.0112 w_k1_INTC_s002
                                 + 0.745646 w_k1_KO_s002
                                 + 0.719439 w_k1_MAT_s002
                                 + 0.862158 w_k1_MSFT_s002
                                 + 0.866358 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.627675 w_k1_ADM_s002
                                 + 0.75423 w_k1_AXP_s002
                                 + 0.851865 w_k1_DIS_s002
                                 + 0.950665 w_k1_FDX_s002
                                 + 0.951952 w_k1_IBM_s002
                                 + 1.08622 w_k1_INTC_s002
                                 + 0.937654 w_k1_KO_s002
                                 + 0.771416 w_k1_MAT_s002
                                 + 1.05544 w_k1_MSFT_s002
                                 + 1.04527 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.690561 w_k1_ADM_s002
                                 + 0.607336 w_k1_AXP_s002
                                 + 0.712177 w_k1_DIS_s002
                                 + 0.772241 w_k1_FDX_s002
                                 + 0.849107 w_k1_IBM_s002
                                 + 1.10953 w_k1_INTC_s002
                                 + 0.99956 w_k1_KO_s002
                                 + 0.717389 w_k1_MAT_s002
                                 + 0.851895 w_k1_MSFT_s002
                                 + 1.00596 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.11529 w_k1_ADM_s002
                                 + 1.34877 w_k1_AXP_s002
                                 + 1.74327 w_k1_DIS_s002
                                 + 1.43196 w_k1_FDX_s002
                                 + 1.15839 w_k1_IBM_s002
                                 + 1.36886 w_k1_INTC_s002
                                 + 1.31202 w_k1_KO_s002 + 1.04161 w_k1_MAT_s002
                                 + 1.26002 w_k1_MSFT_s002
                                 + 1.21932 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.652876 w_k1_ADM_s002
                                 + 0.661242 w_k1_AXP_s002
                                 + 0.636627 w_k1_DIS_s002
                                 + 0.857374 w_k1_FDX_s002
                                 + 0.569932 w_k1_IBM_s002
                                 + 0.935012 w_k1_INTC_s002
                                 + 0.69154 w_k1_KO_s002
                                 + 0.465878 w_k1_MAT_s002
                                 + 0.795996 w_k1_MSFT_s002
                                 + 0.898654 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.762841 w_k1_ADM_s003
                                 + 0.828788 w_k1_AXP_s003
                                 + 0.943648 w_k1_DIS_s003
                                 + 0.840575 w_k1_FDX_s003
                                 + 1.4491 w_k1_IBM_s003
                                 + 1.15862 w_k1_INTC_s003 + 1.1557 w_k1_KO_s003
                                 + 1.02133 w_k1_MAT_s003
                                 + 0.91067 w_k1_MSFT_s003
                                 + 1.00281 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.741935 w_k1_ADM_s003
                                 + 0.977968 w_k1_AXP_s003
                                 + 0.799168 w_k1_DIS_s003
                                 + 0.953134 w_k1_FDX_s003
                                 + 1.00553 w_k1_IBM_s003
                                 + 1.07693 w_k1_INTC_s003
                                 + 0.979108 w_k1_KO_s003
                                 + 1.01231 w_k1_MAT_s003
                                 + 0.942669 w_k1_MSFT_s003
                                 + 0.965275 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.01829 w_k1_ADM_s003
                                 + 0.91487 w_k1_AXP_s003
                                 + 1.18001 w_k1_DIS_s003 + 1.0787 w_k1_FDX_s003
                                 + 1.01861 w_k1_IBM_s003
                                 + 0.937583 w_k1_INTC_s003
                                 + 1.22846 w_k1_KO_s003 + 1.19078 w_k1_MAT_s003
                                 + 1.01405 w_k1_MSFT_s003
                                 + 1.03882 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.02718 w_k1_ADM_s003
                                 + 1.32959 w_k1_AXP_s003
                                 + 1.76813 w_k1_DIS_s003
                                 + 1.50039 w_k1_FDX_s003
                                 + 1.25663 w_k1_IBM_s003
                                 + 1.25251 w_k1_INTC_s003
                                 + 1.29123 w_k1_KO_s003 + 1.40473 w_k1_MAT_s003
                                 + 1.37531 w_k1_MSFT_s003
                                 + 1.33384 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.935053 w_k1_ADM_s003
                                 + 1.07213 w_k1_AXP_s003
                                 + 1.12664 w_k1_DIS_s003
                                 + 1.02742 w_k1_FDX_s003
                                 + 1.38768 w_k1_IBM_s003
                                 + 0.962407 w_k1_INTC_s003
                                 + 1.21974 w_k1_KO_s003 + 1.19949 w_k1_MAT_s003
                                 + 0.970486 w_k1_MSFT_s003
                                 + 1.12445 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.2112 w_k1_ADM_s003
                                 + 1.02047 w_k1_AXP_s003
                                 + 0.906947 w_k1_DIS_s003
                                 + 1.05413 w_k1_FDX_s003
                                 + 0.741335 w_k1_IBM_s003
                                 + 0.974108 w_k1_INTC_s003
                                 + 1.05648 w_k1_KO_s003
                                 + 0.818319 w_k1_MAT_s003
                                 + 1.01974 w_k1_MSFT_s003
                                 + 0.892976 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.618924 w_k1_ADM_s003
                                 + 0.640506 w_k1_AXP_s003
                                 + 0.62061 w_k1_DIS_s003
                                 + 0.822097 w_k1_FDX_s003
                                 + 0.845414 w_k1_IBM_s003
                                 + 0.959984 w_k1_INTC_s003
                                 + 0.881187 w_k1_KO_s003
                                 + 0.719788 w_k1_MAT_s003
                                 + 0.750941 w_k1_MSFT_s003
                                 + 0.804618 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.24773 w_k1_ADM_s003
                                 + 1.38851 w_k1_AXP_s003
                                 + 1.00411 w_k1_DIS_s003 + 1.1859 w_k1_FDX_s003
                                 + 1.1114 w_k1_IBM_s003
                                 + 1.13025 w_k1_INTC_s003
                                 + 1.02983 w_k1_KO_s003 + 1.20632 w_k1_MAT_s003
                                 + 1.07726 w_k1_MSFT_s003
                                 + 1.07549 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.65586 w_k1_ADM_s004
                                 + 1.40195 w_k1_AXP_s004
                                 + 1.62793 w_k1_DIS_s004
                                 + 1.39049 w_k1_FDX_s004
                                 + 1.25246 w_k1_IBM_s004
                                 + 1.35717 w_k1_INTC_s004
                                 + 1.13924 w_k1_KO_s004 + 1.03162 w_k1_MAT_s004
                                 + 1.32282 w_k1_MSFT_s004
                                 + 1.11409 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.06487 w_k1_ADM_s004
                                 + 0.951954 w_k1_AXP_s004
                                 + 0.929524 w_k1_DIS_s004
                                 + 1.17467 w_k1_FDX_s004
                                 + 0.64156 w_k1_IBM_s004
                                 + 1.29845 w_k1_INTC_s004
                                 + 0.880598 w_k1_KO_s004 + 0.4649 w_k1_MAT_s004
                                 + 1.1441 w_k1_MSFT_s004 + 1.09933 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.875989 w_k1_ADM_s004
                                 + 1.02118 w_k1_AXP_s004
                                 + 0.89759 w_k1_DIS_s004
                                 + 0.891148 w_k1_FDX_s004
                                 + 1.06479 w_k1_IBM_s004
                                 + 1.05194 w_k1_INTC_s004 + 0.9842 w_k1_KO_s004
                                 + 1.12792 w_k1_MAT_s004
                                 + 1.05714 w_k1_MSFT_s004
                                 + 1.08727 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.879063 w_k1_ADM_s004
                                 + 0.810936 w_k1_AXP_s004
                                 + 0.763892 w_k1_DIS_s004
                                 + 0.926899 w_k1_FDX_s004
                                 + 0.801715 w_k1_IBM_s004
                                 + 0.931972 w_k1_INTC_s004
                                 + 0.913072 w_k1_KO_s004
                                 + 0.594269 w_k1_MAT_s004
                                 + 0.926759 w_k1_MSFT_s004
                                 + 1.02203 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.21894 w_k1_ADM_s004
                                 + 1.30538 w_k1_AXP_s004
                                 + 1.19415 w_k1_DIS_s004
                                 + 1.21784 w_k1_FDX_s004
                                 + 0.895085 w_k1_IBM_s004
                                 + 1.0066 w_k1_INTC_s004
                                 + 0.975256 w_k1_KO_s004
                                 + 0.934427 w_k1_MAT_s004
                                 + 1.03806 w_k1_MSFT_s004
                                 + 1.01255 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.872682 w_k1_ADM_s004
                                 + 0.929555 w_k1_AXP_s004
                                 + 0.959014 w_k1_DIS_s004
                                 + 0.992953 w_k1_FDX_s004
                                 + 0.89656 w_k1_IBM_s004
                                 + 1.21343 w_k1_INTC_s004
                                 + 0.943952 w_k1_KO_s004
                                 + 0.965414 w_k1_MAT_s004
                                 + 1.02125 w_k1_MSFT_s004
                                 + 1.08369 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.655292 w_k1_ADM_s004
                                 + 0.859573 w_k1_AXP_s004
                                 + 0.889938 w_k1_DIS_s004
                                 + 0.885161 w_k1_FDX_s004
                                 + 1.2931 w_k1_IBM_s004
                                 + 0.936485 w_k1_INTC_s004
                                 + 0.969226 w_k1_KO_s004
                                 + 0.994805 w_k1_MAT_s004
                                 + 0.941342 w_k1_MSFT_s004
                                 + 0.907973 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.84678 w_k1_ADM_s004
                                 + 1.17569 w_k1_AXP_s004
                                 + 1.25893 w_k1_DIS_s004
                                 + 1.17761 w_k1_FDX_s004 + 1.4769 w_k1_IBM_s004
                                 + 1.26743 w_k1_INTC_s004
                                 + 1.21275 w_k1_KO_s004 + 1.14305 w_k1_MAT_s004
                                 + 1.25818 w_k1_MSFT_s004
                                 + 1.26335 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.21891 w_k1_ADM_s005
                                 + 1.09064 w_k1_AXP_s005
                                 + 1.34209 w_k1_DIS_s005
                                 + 1.22205 w_k1_FDX_s005
                                 + 0.855667 w_k1_IBM_s005
                                 + 0.989499 w_k1_INTC_s005
                                 + 1.0592 w_k1_KO_s005 + 0.824389 w_k1_MAT_s005
                                 + 1.1303 w_k1_MSFT_s005 + 1.0052 w_k1_PG_s005
                                 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.17692 w_k1_ADM_s005
                                 + 1.15604 w_k1_AXP_s005
                                 + 0.870987 w_k1_DIS_s005
                                 + 1.04662 w_k1_FDX_s005
                                 + 0.905017 w_k1_IBM_s005
                                 + 0.778602 w_k1_INTC_s005
                                 + 0.940531 w_k1_KO_s005
                                 + 0.934674 w_k1_MAT_s005
                                 + 0.843817 w_k1_MSFT_s005
                                 + 1.06883 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.31698 w_k1_ADM_s005
                                 + 1.31914 w_k1_AXP_s005
                                 + 1.91065 w_k1_DIS_s005
                                 + 1.35108 w_k1_FDX_s005
                                 + 1.38837 w_k1_IBM_s005
                                 + 1.35423 w_k1_INTC_s005
                                 + 1.63416 w_k1_KO_s005 + 1.22028 w_k1_MAT_s005
                                 + 1.62138 w_k1_MSFT_s005
                                 + 1.44669 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.867442 w_k1_ADM_s005
                                 + 0.855482 w_k1_AXP_s005
                                 + 0.787961 w_k1_DIS_s005
                                 + 0.938864 w_k1_FDX_s005
                                 + 0.680131 w_k1_IBM_s005
                                 + 0.846939 w_k1_INTC_s005
                                 + 0.765448 w_k1_KO_s005
                                 + 0.716359 w_k1_MAT_s005
                                 + 0.874067 w_k1_MSFT_s005
                                 + 0.836098 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.5983 w_k1_ADM_s005
                                 + 1.70561 w_k1_AXP_s005
                                 + 1.61763 w_k1_DIS_s005
                                 + 1.48995 w_k1_FDX_s005
                                 + 1.27058 w_k1_IBM_s005
                                 + 1.0399 w_k1_INTC_s005 + 1.20163 w_k1_KO_s005
                                 + 1.24597 w_k1_MAT_s005
                                 + 1.20722 w_k1_MSFT_s005
                                 + 1.16208 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.878963 w_k1_ADM_s005
                                 + 0.918413 w_k1_AXP_s005
                                 + 0.806238 w_k1_DIS_s005
                                 + 0.814706 w_k1_FDX_s005
                                 + 1.01069 w_k1_IBM_s005
                                 + 0.715846 w_k1_INTC_s005
                                 + 0.925291 w_k1_KO_s005
                                 + 0.916977 w_k1_MAT_s005
                                 + 0.776684 w_k1_MSFT_s005
                                 + 0.903221 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.18297 w_k1_ADM_s005
                                 + 1.27622 w_k1_AXP_s005
                                 + 1.24353 w_k1_DIS_s005
                                 + 1.13314 w_k1_FDX_s005 + 1.3877 w_k1_IBM_s005
                                 + 1.11285 w_k1_INTC_s005
                                 + 1.23937 w_k1_KO_s005 + 1.16237 w_k1_MAT_s005
                                 + 1.02758 w_k1_MSFT_s005
                                 + 1.05085 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.904058 w_k1_ADM_s005
                                 + 0.733196 w_k1_AXP_s005
                                 + 0.568627 w_k1_DIS_s005
                                 + 0.716802 w_k1_FDX_s005
                                 + 0.638704 w_k1_IBM_s005
                                 + 0.744613 w_k1_INTC_s005
                                 + 0.695048 w_k1_KO_s005
                                 + 0.503768 w_k1_MAT_s005
                                 + 0.736415 w_k1_MSFT_s005
                                 + 0.82258 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.857015 w_k1_ADM_s006
                                 + 0.597338 w_k1_AXP_s006
                                 + 0.677072 w_k1_DIS_s006
                                 + 0.683106 w_k1_FDX_s006
                                 + 1.07224 w_k1_IBM_s006
                                 + 0.946996 w_k1_INTC_s006
                                 + 1.04476 w_k1_KO_s006
                                 + 0.863769 w_k1_MAT_s006
                                 + 0.98514 w_k1_MSFT_s006
                                 + 0.874138 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.906426 w_k1_ADM_s006
                                 + 0.694495 w_k1_AXP_s006
                                 + 0.785055 w_k1_DIS_s006
                                 + 0.955744 w_k1_FDX_s006
                                 + 0.805961 w_k1_IBM_s006
                                 + 0.998747 w_k1_INTC_s006
                                 + 0.924744 w_k1_KO_s006
                                 + 0.537879 w_k1_MAT_s006
                                 + 1.04948 w_k1_MSFT_s006
                                 + 0.871494 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.953976 w_k1_ADM_s006
                                 + 0.961867 w_k1_AXP_s006
                                 + 1.18927 w_k1_DIS_s006
                                 + 1.02209 w_k1_FDX_s006
                                 + 1.33724 w_k1_IBM_s006
                                 + 1.11581 w_k1_INTC_s006 + 1.2996 w_k1_KO_s006
                                 + 0.975146 w_k1_MAT_s006
                                 + 1.12935 w_k1_MSFT_s006
                                 + 1.08396 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.37984 w_k1_ADM_s006
                                 + 1.46931 w_k1_AXP_s006
                                 + 1.75041 w_k1_DIS_s006
                                 + 1.49043 w_k1_FDX_s006
                                 + 1.52559 w_k1_IBM_s006
                                 + 1.36874 w_k1_INTC_s006
                                 + 1.48277 w_k1_KO_s006 + 1.79518 w_k1_MAT_s006
                                 + 1.44572 w_k1_MSFT_s006
                                 + 1.32283 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.20024 w_k1_ADM_s006
                                 + 0.911425 w_k1_AXP_s006
                                 + 0.926409 w_k1_DIS_s006
                                 + 1.09983 w_k1_FDX_s006
                                 + 0.713232 w_k1_IBM_s006
                                 + 0.914338 w_k1_INTC_s006
                                 + 0.785357 w_k1_KO_s006
                                 + 0.670148 w_k1_MAT_s006
                                 + 0.968209 w_k1_MSFT_s006
                                 + 0.937982 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.859792 w_k1_ADM_s006
                                 + 1.04411 w_k1_AXP_s006
                                 + 1.08353 w_k1_DIS_s006
                                 + 0.877804 w_k1_FDX_s006
                                 + 1.73895 w_k1_IBM_s006
                                 + 1.11588 w_k1_INTC_s006
                                 + 1.56889 w_k1_KO_s006 + 2.06259 w_k1_MAT_s006
                                 + 1.15081 w_k1_MSFT_s006
                                 + 1.14758 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.848073 w_k1_ADM_s006
                                 + 0.566539 w_k1_AXP_s006
                                 + 0.592455 w_k1_DIS_s006
                                 + 0.782786 w_k1_FDX_s006
                                 + 0.628927 w_k1_IBM_s006
                                 + 0.833339 w_k1_INTC_s006
                                 + 0.782145 w_k1_KO_s006
                                 + 0.49127 w_k1_MAT_s006
                                 + 0.976773 w_k1_MSFT_s006
                                 + 0.824111 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.04051 w_k1_ADM_s006
                                 + 0.822394 w_k1_AXP_s006
                                 + 0.873753 w_k1_DIS_s006
                                 + 0.944502 w_k1_FDX_s006
                                 + 1.17767 w_k1_IBM_s006
                                 + 1.11355 w_k1_INTC_s006
                                 + 1.04157 w_k1_KO_s006
                                 + 0.813094 w_k1_MAT_s006
                                 + 1.0301 w_k1_MSFT_s006 + 1.12646 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.08257 w_k1_ADM_s007
                                 + 0.783539 w_k1_AXP_s007
                                 + 0.914398 w_k1_DIS_s007
                                 + 0.967435 w_k1_FDX_s007
                                 + 0.704334 w_k1_IBM_s007
                                 + 0.920131 w_k1_INTC_s007
                                 + 0.864289 w_k1_KO_s007
                                 + 0.637714 w_k1_MAT_s007
                                 + 0.89119 w_k1_MSFT_s007
                                 + 0.913827 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.27475 w_k1_ADM_s007
                                 + 1.27053 w_k1_AXP_s007 + 1.1538 w_k1_DIS_s007
                                 + 1.05559 w_k1_FDX_s007
                                 + 1.57505 w_k1_IBM_s007
                                 + 0.930829 w_k1_INTC_s007
                                 + 1.14258 w_k1_KO_s007 + 1.19785 w_k1_MAT_s007
                                 + 0.860385 w_k1_MSFT_s007
                                 + 1.07187 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.2762 w_k1_ADM_s007
                                 + 1.33233 w_k1_AXP_s007
                                 + 1.54638 w_k1_DIS_s007
                                 + 1.25418 w_k1_FDX_s007
                                 + 1.10768 w_k1_IBM_s007
                                 + 1.38954 w_k1_INTC_s007
                                 + 1.42778 w_k1_KO_s007 + 1.15607 w_k1_MAT_s007
                                 + 1.23623 w_k1_MSFT_s007
                                 + 1.05934 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.23076 w_k1_ADM_s007
                                 + 0.930828 w_k1_AXP_s007
                                 + 1.05578 w_k1_DIS_s007
                                 + 0.949949 w_k1_FDX_s007
                                 + 1.08854 w_k1_IBM_s007
                                 + 0.875794 w_k1_INTC_s007
                                 + 1.05253 w_k1_KO_s007
                                 + 0.871968 w_k1_MAT_s007
                                 + 0.88865 w_k1_MSFT_s007
                                 + 0.82792 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.959706 w_k1_ADM_s007
                                 + 0.989416 w_k1_AXP_s007
                                 + 0.790011 w_k1_DIS_s007
                                 + 0.845359 w_k1_FDX_s007
                                 + 0.905214 w_k1_IBM_s007
                                 + 0.970789 w_k1_INTC_s007
                                 + 0.8489 w_k1_KO_s007 + 0.602794 w_k1_MAT_s007
                                 + 0.796328 w_k1_MSFT_s007
                                 + 0.750499 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.25815 w_k1_ADM_s007
                                 + 1.24341 w_k1_AXP_s007
                                 + 0.987967 w_k1_DIS_s007
                                 + 1.07667 w_k1_FDX_s007
                                 + 0.820303 w_k1_IBM_s007
                                 + 0.938222 w_k1_INTC_s007
                                 + 0.948631 w_k1_KO_s007
                                 + 0.961023 w_k1_MAT_s007
                                 + 0.983582 w_k1_MSFT_s007
                                 + 0.84805 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.15946 w_k1_ADM_s007
                                 + 1.08124 w_k1_AXP_s007
                                 + 1.20771 w_k1_DIS_s007
                                 + 1.17474 w_k1_FDX_s007
                                 + 1.22381 w_k1_IBM_s007
                                 + 1.15373 w_k1_INTC_s007
                                 + 1.24721 w_k1_KO_s007
                                 + 0.860746 w_k1_MAT_s007
                                 + 1.10473 w_k1_MSFT_s007
                                 + 1.10728 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.682022 w_k1_ADM_s007
                                 + 0.73 w_k1_AXP_s007 + 0.812949 w_k1_DIS_s007
                                 + 0.772921 w_k1_FDX_s007
                                 + 1.17014 w_k1_IBM_s007
                                 + 1.05041 w_k1_INTC_s007
                                 + 1.05508 w_k1_KO_s007
                                 + 0.929294 w_k1_MAT_s007
                                 + 0.84248 w_k1_MSFT_s007
                                 + 0.82439 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.933285 w_k1_ADM_s010
                                 + 0.897398 w_k1_AXP_s010
                                 + 0.885006 w_k1_DIS_s010
                                 + 0.937735 w_k1_FDX_s010
                                 + 0.786733 w_k1_IBM_s010
                                 + 0.88043 w_k1_INTC_s010
                                 + 0.87687 w_k1_KO_s010
                                 + 0.785179 w_k1_MAT_s010
                                 + 0.943832 w_k1_MSFT_s010
                                 + 0.800562 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.894189 w_k1_ADM_s010
                                 + 0.879979 w_k1_AXP_s010
                                 + 0.880798 w_k1_DIS_s010
                                 + 0.931899 w_k1_FDX_s010
                                 + 1.12872 w_k1_IBM_s010
                                 + 0.895344 w_k1_INTC_s010
                                 + 1.00588 w_k1_KO_s010 + 1.17109 w_k1_MAT_s010
                                 + 0.963106 w_k1_MSFT_s010
                                 + 0.921461 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.43031 w_k1_ADM_s010
                                 + 1.18832 w_k1_AXP_s010
                                 + 1.29929 w_k1_DIS_s010
                                 + 1.14983 w_k1_FDX_s010
                                 + 1.10177 w_k1_IBM_s010
                                 + 1.06841 w_k1_INTC_s010
                                 + 1.40556 w_k1_KO_s010 + 1.18388 w_k1_MAT_s010
                                 + 1.23722 w_k1_MSFT_s010
                                 + 1.06431 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.760965 w_k1_ADM_s010
                                 + 0.965079 w_k1_AXP_s010
                                 + 1.12314 w_k1_DIS_s010
                                 + 0.983162 w_k1_FDX_s010
                                 + 1.44584 w_k1_IBM_s010
                                 + 1.05194 w_k1_INTC_s010
                                 + 1.37455 w_k1_KO_s010 + 1.60127 w_k1_MAT_s010
                                 + 1.09469 w_k1_MSFT_s010
                                 + 1.01479 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.13287 w_k1_ADM_s010
                                 + 1.47685 w_k1_AXP_s010
                                 + 1.51125 w_k1_DIS_s010
                                 + 1.28451 w_k1_FDX_s010
                                 + 1.51273 w_k1_IBM_s010
                                 + 1.25749 w_k1_INTC_s010
                                 + 1.43458 w_k1_KO_s010 + 1.89481 w_k1_MAT_s010
                                 + 1.26128 w_k1_MSFT_s010
                                 + 1.10167 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.0196 w_k1_ADM_s010
                                 + 1.07946 w_k1_AXP_s010
                                 + 1.17152 w_k1_DIS_s010
                                 + 1.15883 w_k1_FDX_s010
                                 + 0.880261 w_k1_IBM_s010
                                 + 1.03105 w_k1_INTC_s010
                                 + 0.993645 w_k1_KO_s010
                                 + 0.951612 w_k1_MAT_s010
                                 + 1.07371 w_k1_MSFT_s010
                                 + 1.03989 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.29021 w_k1_ADM_s010
                                 + 1.28852 w_k1_AXP_s010
                                 + 1.78831 w_k1_DIS_s010
                                 + 1.34829 w_k1_FDX_s010
                                 + 1.17586 w_k1_IBM_s010
                                 + 1.4221 w_k1_INTC_s010 + 1.50721 w_k1_KO_s010
                                 + 1.12461 w_k1_MAT_s010
                                 + 1.39673 w_k1_MSFT_s010
                                 + 1.29615 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.685526 w_k1_ADM_s010
                                 + 0.791331 w_k1_AXP_s010
                                 + 0.641588 w_k1_DIS_s010
                                 + 0.790145 w_k1_FDX_s010
                                 + 0.716062 w_k1_IBM_s010
                                 + 0.764638 w_k1_INTC_s010
                                 + 0.730005 w_k1_KO_s010
                                 + 0.989466 w_k1_MAT_s010
                                 + 0.857979 w_k1_MSFT_s010
                                 + 0.767014 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.00185 w_k1_ADM_s011
                                 + 1.07927 w_k1_AXP_s011
                                 + 0.869538 w_k1_DIS_s011
                                 + 1.03342 w_k1_FDX_s011
                                 + 0.74419 w_k1_IBM_s011
                                 + 1.02647 w_k1_INTC_s011
                                 + 0.875519 w_k1_KO_s011
                                 + 0.836198 w_k1_MAT_s011
                                 + 0.948202 w_k1_MSFT_s011
                                 + 1.05002 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.04397 w_k1_ADM_s011
                                 + 1.15079 w_k1_AXP_s011
                                 + 1.07036 w_k1_DIS_s011
                                 + 1.00779 w_k1_FDX_s011
                                 + 1.39754 w_k1_IBM_s011
                                 + 1.13882 w_k1_INTC_s011
                                 + 1.12362 w_k1_KO_s011 + 1.28924 w_k1_MAT_s011
                                 + 0.991694 w_k1_MSFT_s011
                                 + 1.10784 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.783582 w_k1_ADM_s011
                                 + 0.76083 w_k1_AXP_s011
                                 + 0.742602 w_k1_DIS_s011
                                 + 0.769339 w_k1_FDX_s011
                                 + 1.04903 w_k1_IBM_s011
                                 + 0.932248 w_k1_INTC_s011
                                 + 0.964754 w_k1_KO_s011
                                 + 1.23098 w_k1_MAT_s011
                                 + 0.804258 w_k1_MSFT_s011
                                 + 1.00194 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.23546 w_k1_ADM_s011
                                 + 1.24083 w_k1_AXP_s011
                                 + 1.33001 w_k1_DIS_s011 + 1.1925 w_k1_FDX_s011
                                 + 1.30668 w_k1_IBM_s011
                                 + 1.51182 w_k1_INTC_s011
                                 + 1.21244 w_k1_KO_s011 + 1.58006 w_k1_MAT_s011
                                 + 1.43138 w_k1_MSFT_s011
                                 + 1.14593 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.21567 w_k1_ADM_s011
                                 + 1.49697 w_k1_AXP_s011
                                 + 1.47008 w_k1_DIS_s011
                                 + 1.37186 w_k1_FDX_s011
                                 + 0.936061 w_k1_IBM_s011
                                 + 1.30818 w_k1_INTC_s011
                                 + 1.13873 w_k1_KO_s011 + 1.10034 w_k1_MAT_s011
                                 + 1.44353 w_k1_MSFT_s011
                                 + 1.25548 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.888318 w_k1_ADM_s011
                                 + 0.76775 w_k1_AXP_s011
                                 + 0.653393 w_k1_DIS_s011
                                 + 0.913265 w_k1_FDX_s011
                                 + 0.70441 w_k1_IBM_s011
                                 + 1.05263 w_k1_INTC_s011
                                 + 0.747726 w_k1_KO_s011
                                 + 0.693892 w_k1_MAT_s011
                                 + 0.912323 w_k1_MSFT_s011
                                 + 0.942796 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.962389 w_k1_ADM_s011
                                 + 1.08691 w_k1_AXP_s011
                                 + 1.12692 w_k1_DIS_s011
                                 + 1.01433 w_k1_FDX_s011
                                 + 0.978095 w_k1_IBM_s011
                                 + 0.951722 w_k1_INTC_s011
                                 + 0.930882 w_k1_KO_s011
                                 + 0.940037 w_k1_MAT_s011
                                 + 1.09883 w_k1_MSFT_s011
                                 + 0.998497 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.18589 w_k1_ADM_s011
                                 + 1.38608 w_k1_AXP_s011
                                 + 1.10339 w_k1_DIS_s011
                                 + 1.25245 w_k1_FDX_s011
                                 + 0.815782 w_k1_IBM_s011
                                 + 1.19319 w_k1_INTC_s011
                                 + 0.911648 w_k1_KO_s011
                                 + 0.864276 w_k1_MAT_s011
                                 + 1.09853 w_k1_MSFT_s011
                                 + 1.16692 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.647299 w_k1_ADM_s012
                                 + 0.744304 w_k1_AXP_s012
                                 + 0.667431 w_k1_DIS_s012
                                 + 0.799076 w_k1_FDX_s012
                                 + 0.861524 w_k1_IBM_s012
                                 + 0.927002 w_k1_INTC_s012
                                 + 0.821999 w_k1_KO_s012
                                 + 0.894104 w_k1_MAT_s012
                                 + 0.854521 w_k1_MSFT_s012
                                 + 0.884554 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.866664 w_k1_ADM_s012
                                 + 1.23176 w_k1_AXP_s012
                                 + 0.969589 w_k1_DIS_s012
                                 + 1.0358 w_k1_FDX_s012 + 1.0394 w_k1_IBM_s012
                                 + 1.07818 w_k1_INTC_s012
                                 + 1.04309 w_k1_KO_s012 + 1.29801 w_k1_MAT_s012
                                 + 1.07985 w_k1_MSFT_s012
                                 + 1.17454 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.19733 w_k1_ADM_s012
                                 + 1.12339 w_k1_AXP_s012
                                 + 1.41211 w_k1_DIS_s012
                                 + 1.20348 w_k1_FDX_s012
                                 + 1.44703 w_k1_IBM_s012
                                 + 1.16943 w_k1_INTC_s012
                                 + 1.19032 w_k1_KO_s012 + 1.24326 w_k1_MAT_s012
                                 + 1.31289 w_k1_MSFT_s012
                                 + 1.29745 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.803397 w_k1_ADM_s012
                                 + 0.809634 w_k1_AXP_s012
                                 + 0.877181 w_k1_DIS_s012
                                 + 0.942267 w_k1_FDX_s012
                                 + 0.972889 w_k1_IBM_s012
                                 + 1.10354 w_k1_INTC_s012
                                 + 0.921822 w_k1_KO_s012
                                 + 0.773851 w_k1_MAT_s012
                                 + 1.02424 w_k1_MSFT_s012
                                 + 1.02874 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.762633 w_k1_ADM_s012
                                 + 0.697552 w_k1_AXP_s012
                                 + 0.846055 w_k1_DIS_s012
                                 + 0.790389 w_k1_FDX_s012
                                 + 1.21331 w_k1_IBM_s012
                                 + 0.952959 w_k1_INTC_s012
                                 + 0.918274 w_k1_KO_s012
                                 + 0.744489 w_k1_MAT_s012
                                 + 0.837699 w_k1_MSFT_s012
                                 + 0.978025 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.681568 w_k1_ADM_s012
                                 + 0.990828 w_k1_AXP_s012
                                 + 0.978121 w_k1_DIS_s012
                                 + 1.10631 w_k1_FDX_s012
                                 + 0.668704 w_k1_IBM_s012
                                 + 1.00082 w_k1_INTC_s012
                                 + 0.735248 w_k1_KO_s012
                                 + 0.815406 w_k1_MAT_s012
                                 + 0.976421 w_k1_MSFT_s012
                                 + 1.08231 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.949352 w_k1_ADM_s012
                                 + 1.064 w_k1_AXP_s012 + 1.24753 w_k1_DIS_s012
                                 + 1.17922 w_k1_FDX_s012
                                 + 0.997984 w_k1_IBM_s012
                                 + 1.07343 w_k1_INTC_s012
                                 + 1.05827 w_k1_KO_s012 + 0.89557 w_k1_MAT_s012
                                 + 1.19013 w_k1_MSFT_s012 + 1.1535 w_k1_PG_s012
                                 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.789453 w_k1_ADM_s012
                                 + 0.60234 w_k1_AXP_s012
                                 + 0.563007 w_k1_DIS_s012
                                 + 0.78056 w_k1_FDX_s012
                                 + 0.643335 w_k1_IBM_s012
                                 + 0.822819 w_k1_INTC_s012
                                 + 0.621106 w_k1_KO_s012
                                 + 0.547548 w_k1_MAT_s012
                                 + 0.835598 w_k1_MSFT_s012
                                 + 0.874441 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.980915 w_k1_ADM_s013
                                 + 0.927086 w_k1_AXP_s013
                                 + 0.692532 w_k1_DIS_s013
                                 + 0.760971 w_k1_FDX_s013
                                 + 1.07757 w_k1_IBM_s013
                                 + 0.982896 w_k1_INTC_s013
                                 + 0.90812 w_k1_KO_s013 + 1.26653 w_k1_MAT_s013
                                 + 0.859684 w_k1_MSFT_s013
                                 + 0.92576 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.46919 w_k1_ADM_s013
                                 + 0.874475 w_k1_AXP_s013
                                 + 0.74883 w_k1_DIS_s013
                                 + 1.00465 w_k1_FDX_s013
                                 + 0.635783 w_k1_IBM_s013
                                 + 0.998112 w_k1_INTC_s013
                                 + 0.785074 w_k1_KO_s013
                                 + 0.669178 w_k1_MAT_s013
                                 + 0.86766 w_k1_MSFT_s013
                                 + 0.865531 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.41873 w_k1_ADM_s013
                                 + 1.15224 w_k1_AXP_s013
                                 + 1.33269 w_k1_DIS_s013
                                 + 1.03769 w_k1_FDX_s013
                                 + 1.71913 w_k1_IBM_s013
                                 + 0.860981 w_k1_INTC_s013
                                 + 1.29197 w_k1_KO_s013 + 1.98582 w_k1_MAT_s013
                                 + 0.945536 w_k1_MSFT_s013
                                 + 1.12342 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.49167 w_k1_ADM_s013
                                 + 1.22742 w_k1_AXP_s013
                                 + 1.05947 w_k1_DIS_s013
                                 + 1.11934 w_k1_FDX_s013
                                 + 0.829683 w_k1_IBM_s013
                                 + 0.956329 w_k1_INTC_s013
                                 + 0.907586 w_k1_KO_s013
                                 + 1.29619 w_k1_MAT_s013
                                 + 0.979295 w_k1_MSFT_s013
                                 + 0.969941 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.47732 w_k1_ADM_s013
                                 + 1.48962 w_k1_AXP_s013
                                 + 1.23727 w_k1_DIS_s013
                                 + 1.28361 w_k1_FDX_s013
                                 + 1.07911 w_k1_IBM_s013
                                 + 1.04796 w_k1_INTC_s013
                                 + 1.10432 w_k1_KO_s013 + 1.47899 w_k1_MAT_s013
                                 + 0.918629 w_k1_MSFT_s013
                                 + 1.04187 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.38381 w_k1_ADM_s013
                                 + 1.19097 w_k1_AXP_s013
                                 + 1.16825 w_k1_DIS_s013
                                 + 1.00425 w_k1_FDX_s013
                                 + 1.29779 w_k1_IBM_s013
                                 + 0.924012 w_k1_INTC_s013
                                 + 1.15021 w_k1_KO_s013 + 1.40607 w_k1_MAT_s013
                                 + 0.966482 w_k1_MSFT_s013
                                 + 1.07804 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.03284 w_k1_ADM_s013
                                 + 0.835137 w_k1_AXP_s013
                                 + 0.990997 w_k1_DIS_s013
                                 + 0.955934 w_k1_FDX_s013
                                 + 0.92059 w_k1_IBM_s013
                                 + 0.969177 w_k1_INTC_s013
                                 + 0.884058 w_k1_KO_s013
                                 + 0.898293 w_k1_MAT_s013
                                 + 0.945574 w_k1_MSFT_s013
                                 + 0.942454 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.44264 w_k1_ADM_s013
                                 + 1.06416 w_k1_AXP_s013
                                 + 1.45507 w_k1_DIS_s013
                                 + 1.33279 w_k1_FDX_s013
                                 + 1.06444 w_k1_IBM_s013
                                 + 1.36766 w_k1_INTC_s013
                                 + 1.07089 w_k1_KO_s013 + 1.02257 w_k1_MAT_s013
                                 + 1.2033 w_k1_MSFT_s013 + 1.10496 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.81461 w_k1_ADM_s014
                                 + 0.60864 w_k1_AXP_s014
                                 + 0.648083 w_k1_DIS_s014
                                 + 0.704058 w_k1_FDX_s014
                                 + 0.721368 w_k1_IBM_s014
                                 + 0.532294 w_k1_INTC_s014
                                 + 0.690338 w_k1_KO_s014
                                 + 0.798824 w_k1_MAT_s014
                                 + 0.744496 w_k1_MSFT_s014
                                 + 0.779389 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.35644 w_k1_ADM_s014
                                 + 1.0083 w_k1_AXP_s014 + 1.43135 w_k1_DIS_s014
                                 + 1.1628 w_k1_FDX_s014 + 0.82706 w_k1_IBM_s014
                                 + 0.909536 w_k1_INTC_s014
                                 + 1.09831 w_k1_KO_s014 + 1.13104 w_k1_MAT_s014
                                 + 1.0893 w_k1_MSFT_s014 + 1.11391 w_k1_PG_s014
                                 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.26805 w_k1_ADM_s014
                                 + 1.21425 w_k1_AXP_s014
                                 + 0.991102 w_k1_DIS_s014
                                 + 0.972434 w_k1_FDX_s014
                                 + 1.18073 w_k1_IBM_s014
                                 + 0.865943 w_k1_INTC_s014
                                 + 1.01995 w_k1_KO_s014 + 1.63556 w_k1_MAT_s014
                                 + 0.905945 w_k1_MSFT_s014
                                 + 1.06647 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.03624 w_k1_ADM_s014
                                 + 1.49016 w_k1_AXP_s014
                                 + 1.21655 w_k1_DIS_s014
                                 + 1.14775 w_k1_FDX_s014
                                 + 1.08575 w_k1_IBM_s014
                                 + 1.18613 w_k1_INTC_s014 + 1.2058 w_k1_KO_s014
                                 + 1.25967 w_k1_MAT_s014
                                 + 1.10464 w_k1_MSFT_s014
                                 + 1.09369 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.02538 w_k1_ADM_s014
                                 + 0.721523 w_k1_AXP_s014
                                 + 0.847876 w_k1_DIS_s014
                                 + 0.803384 w_k1_FDX_s014
                                 + 1.15359 w_k1_IBM_s014 + 0.807 w_k1_INTC_s014
                                 + 1.22873 w_k1_KO_s014 + 1.16089 w_k1_MAT_s014
                                 + 0.841736 w_k1_MSFT_s014
                                 + 0.979482 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.819854 w_k1_ADM_s014
                                 + 0.907175 w_k1_AXP_s014
                                 + 0.766289 w_k1_DIS_s014
                                 + 0.875269 w_k1_FDX_s014
                                 + 0.862641 w_k1_IBM_s014
                                 + 0.82998 w_k1_INTC_s014
                                 + 0.766358 w_k1_KO_s014
                                 + 0.909293 w_k1_MAT_s014
                                 + 0.768257 w_k1_MSFT_s014
                                 + 0.878112 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.959874 w_k1_ADM_s014
                                 + 1.2219 w_k1_AXP_s014 + 1.43709 w_k1_DIS_s014
                                 + 0.982822 w_k1_FDX_s014
                                 + 1.38175 w_k1_IBM_s014
                                 + 0.935244 w_k1_INTC_s014
                                 + 1.34317 w_k1_KO_s014 + 1.23165 w_k1_MAT_s014
                                 + 0.995529 w_k1_MSFT_s014
                                 + 1.07504 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.836057 w_k1_ADM_s014
                                 + 1.04047 w_k1_AXP_s014
                                 + 0.99337 w_k1_DIS_s014
                                 + 1.02319 w_k1_FDX_s014
                                 + 0.757798 w_k1_IBM_s014
                                 + 0.922321 w_k1_INTC_s014
                                 + 0.887773 w_k1_KO_s014
                                 + 0.818538 w_k1_MAT_s014
                                 + 0.864147 w_k1_MSFT_s014
                                 + 0.885946 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.893367 w_k1_ADM_s015
                                 + 0.646929 w_k1_AXP_s015
                                 + 0.643103 w_k1_DIS_s015
                                 + 0.714315 w_k1_FDX_s015
                                 + 0.851555 w_k1_IBM_s015
                                 + 0.879139 w_k1_INTC_s015
                                 + 0.731974 w_k1_KO_s015
                                 + 0.774233 w_k1_MAT_s015
                                 + 0.789636 w_k1_MSFT_s015
                                 + 0.80551 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.13812 w_k1_ADM_s015
                                 + 0.787946 w_k1_AXP_s015
                                 + 0.971624 w_k1_DIS_s015
                                 + 0.899448 w_k1_FDX_s015
                                 + 0.728671 w_k1_IBM_s015
                                 + 0.97693 w_k1_INTC_s015
                                 + 0.850868 w_k1_KO_s015
                                 + 0.728981 w_k1_MAT_s015
                                 + 1.00044 w_k1_MSFT_s015
                                 + 0.804861 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.790365 w_k1_ADM_s015
                                 + 0.695241 w_k1_AXP_s015
                                 + 0.719895 w_k1_DIS_s015
                                 + 0.72715 w_k1_FDX_s015
                                 + 1.26171 w_k1_IBM_s015
                                 + 0.658227 w_k1_INTC_s015
                                 + 0.858688 w_k1_KO_s015
                                 + 1.21483 w_k1_MAT_s015
                                 + 0.680022 w_k1_MSFT_s015
                                 + 0.960179 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.11191 w_k1_ADM_s015
                                 + 0.784129 w_k1_AXP_s015
                                 + 0.659634 w_k1_DIS_s015
                                 + 0.86872 w_k1_FDX_s015
                                 + 0.648732 w_k1_IBM_s015
                                 + 0.90133 w_k1_INTC_s015
                                 + 0.670302 w_k1_KO_s015
                                 + 0.801924 w_k1_MAT_s015
                                 + 0.850176 w_k1_MSFT_s015
                                 + 0.925168 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.19729 w_k1_ADM_s015
                                 + 0.710408 w_k1_AXP_s015
                                 + 1.06076 w_k1_DIS_s015
                                 + 0.969594 w_k1_FDX_s015
                                 + 0.925037 w_k1_IBM_s015
                                 + 0.818986 w_k1_INTC_s015
                                 + 1.11856 w_k1_KO_s015
                                 + 0.920435 w_k1_MAT_s015
                                 + 1.13761 w_k1_MSFT_s015
                                 + 0.846568 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.61838 w_k1_ADM_s015
                                 + 1.1043 w_k1_AXP_s015 + 1.25251 w_k1_DIS_s015
                                 + 1.28626 w_k1_FDX_s015
                                 + 0.775137 w_k1_IBM_s015
                                 + 0.920543 w_k1_INTC_s015
                                 + 0.929214 w_k1_KO_s015
                                 + 0.938271 w_k1_MAT_s015
                                 + 1.17193 w_k1_MSFT_s015
                                 + 1.20987 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.3048 w_k1_ADM_s015
                                 + 1.35403 w_k1_AXP_s015 + 1.6577 w_k1_DIS_s015
                                 + 1.16124 w_k1_FDX_s015
                                 + 1.32785 w_k1_IBM_s015
                                 + 1.20564 w_k1_INTC_s015 + 1.3014 w_k1_KO_s015
                                 + 1.59362 w_k1_MAT_s015
                                 + 1.23731 w_k1_MSFT_s015
                                 + 1.22805 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.23986 w_k1_ADM_s015
                                 + 1.04849 w_k1_AXP_s015
                                 + 0.880782 w_k1_DIS_s015
                                 + 0.982304 w_k1_FDX_s015
                                 + 1.11653 w_k1_IBM_s015
                                 + 0.796171 w_k1_INTC_s015
                                 + 0.938485 w_k1_KO_s015
                                 + 1.32889 w_k1_MAT_s015
                                 + 0.88388 w_k1_MSFT_s015
                                 + 1.02144 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.299 w_k1_ADM_s016
                                 + 1.21819 w_k1_AXP_s016
                                 + 1.46889 w_k1_DIS_s016
                                 + 1.29892 w_k1_FDX_s016
                                 + 1.27916 w_k1_IBM_s016
                                 + 1.25999 w_k1_INTC_s016
                                 + 1.58607 w_k1_KO_s016 + 1.40622 w_k1_MAT_s016
                                 + 1.3588 w_k1_MSFT_s016 + 1.14002 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.951423 w_k1_ADM_s016
                                 + 0.686709 w_k1_AXP_s016
                                 + 1.00849 w_k1_DIS_s016
                                 + 0.930092 w_k1_FDX_s016
                                 + 1.13317 w_k1_IBM_s016
                                 + 0.935044 w_k1_INTC_s016
                                 + 1.11404 w_k1_KO_s016 + 1.03457 w_k1_MAT_s016
                                 + 1.21965 w_k1_MSFT_s016
                                 + 1.00488 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.880995 w_k1_ADM_s016
                                 + 0.980378 w_k1_AXP_s016
                                 + 1.06705 w_k1_DIS_s016
                                 + 1.03275 w_k1_FDX_s016
                                 + 1.01144 w_k1_IBM_s016
                                 + 1.0583 w_k1_INTC_s016 + 1.02227 w_k1_KO_s016
                                 + 1.19463 w_k1_MAT_s016
                                 + 0.945248 w_k1_MSFT_s016
                                 + 0.934887 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.09561 w_k1_ADM_s016
                                 + 0.82812 w_k1_AXP_s016
                                 + 0.804217 w_k1_DIS_s016
                                 + 0.944633 w_k1_FDX_s016
                                 + 0.686811 w_k1_IBM_s016
                                 + 0.905075 w_k1_INTC_s016
                                 + 0.874508 w_k1_KO_s016
                                 + 0.704782 w_k1_MAT_s016
                                 + 0.860072 w_k1_MSFT_s016
                                 + 0.848551 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.887644 w_k1_ADM_s016
                                 + 1.08247 w_k1_AXP_s016
                                 + 1.11794 w_k1_DIS_s016
                                 + 0.986326 w_k1_FDX_s016
                                 + 1.58082 w_k1_IBM_s016
                                 + 0.971074 w_k1_INTC_s016
                                 + 1.29818 w_k1_KO_s016 + 2.13072 w_k1_MAT_s016
                                 + 0.917108 w_k1_MSFT_s016
                                 + 1.03272 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.815337 w_k1_ADM_s016
                                 + 0.665197 w_k1_AXP_s016
                                 + 0.609035 w_k1_DIS_s016
                                 + 0.729837 w_k1_FDX_s016
                                 + 0.9055 w_k1_IBM_s016
                                 + 0.724164 w_k1_INTC_s016
                                 + 0.731263 w_k1_KO_s016
                                 + 0.79515 w_k1_MAT_s016
                                 + 0.743155 w_k1_MSFT_s016
                                 + 0.77197 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.64786 w_k1_ADM_s016
                                 + 1.08703 w_k1_AXP_s016
                                 + 1.30703 w_k1_DIS_s016
                                 + 1.32707 w_k1_FDX_s016
                                 + 0.908783 w_k1_IBM_s016
                                 + 1.11055 w_k1_INTC_s016
                                 + 1.09794 w_k1_KO_s016
                                 + 0.794799 w_k1_MAT_s016
                                 + 1.03125 w_k1_MSFT_s016
                                 + 1.07766 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.08154 w_k1_ADM_s016
                                 + 0.697089 w_k1_AXP_s016
                                 + 0.751762 w_k1_DIS_s016
                                 + 0.869111 w_k1_FDX_s016
                                 + 1.31988 w_k1_IBM_s016
                                 + 0.747118 w_k1_INTC_s016
                                 + 0.951744 w_k1_KO_s016
                                 + 1.04055 w_k1_MAT_s016
                                 + 0.81439 w_k1_MSFT_s016
                                 + 0.895525 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.00368 w_k1_ADM_s017
                                 + 0.830997 w_k1_AXP_s017
                                 + 0.903571 w_k1_DIS_s017
                                 + 1.04989 w_k1_FDX_s017
                                 + 0.733733 w_k1_IBM_s017
                                 + 0.901708 w_k1_INTC_s017
                                 + 0.769587 w_k1_KO_s017
                                 + 0.840888 w_k1_MAT_s017
                                 + 0.91998 w_k1_MSFT_s017
                                 + 0.949186 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.949613 w_k1_ADM_s017
                                 + 1.27559 w_k1_AXP_s017
                                 + 1.46396 w_k1_DIS_s017
                                 + 1.32562 w_k1_FDX_s017
                                 + 1.00915 w_k1_IBM_s017
                                 + 1.26057 w_k1_INTC_s017
                                 + 1.46558 w_k1_KO_s017 + 1.27862 w_k1_MAT_s017
                                 + 1.43343 w_k1_MSFT_s017
                                 + 1.20498 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.938876 w_k1_ADM_s017
                                 + 1.08818 w_k1_AXP_s017
                                 + 1.09642 w_k1_DIS_s017
                                 + 0.998157 w_k1_FDX_s017
                                 + 1.28636 w_k1_IBM_s017
                                 + 1.01635 w_k1_INTC_s017
                                 + 1.10255 w_k1_KO_s017 + 1.47974 w_k1_MAT_s017
                                 + 0.989108 w_k1_MSFT_s017
                                 + 1.15966 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.158 w_k1_ADM_s017
                                 + 1.38311 w_k1_AXP_s017
                                 + 1.26485 w_k1_DIS_s017
                                 + 1.25199 w_k1_FDX_s017
                                 + 1.73342 w_k1_IBM_s017
                                 + 1.06737 w_k1_INTC_s017
                                 + 1.18913 w_k1_KO_s017 + 1.9797 w_k1_MAT_s017
                                 + 0.956479 w_k1_MSFT_s017
                                 + 1.18121 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.15501 w_k1_ADM_s017
                                 + 0.997967 w_k1_AXP_s017
                                 + 1.38828 w_k1_DIS_s017
                                 + 1.05718 w_k1_FDX_s017
                                 + 1.07762 w_k1_IBM_s017
                                 + 0.854234 w_k1_INTC_s017
                                 + 1.0111 w_k1_KO_s017 + 1.11476 w_k1_MAT_s017
                                 + 1.01712 w_k1_MSFT_s017
                                 + 1.10951 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.749706 w_k1_ADM_s017
                                 + 1.02758 w_k1_AXP_s017
                                 + 0.653186 w_k1_DIS_s017
                                 + 0.795099 w_k1_FDX_s017
                                 + 0.776775 w_k1_IBM_s017
                                 + 0.84844 w_k1_INTC_s017
                                 + 0.710117 w_k1_KO_s017
                                 + 1.04951 w_k1_MAT_s017
                                 + 0.830812 w_k1_MSFT_s017
                                 + 0.939246 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.02401 w_k1_ADM_s017
                                 + 1.39689 w_k1_AXP_s017
                                 + 1.09895 w_k1_DIS_s017
                                 + 1.06081 w_k1_FDX_s017
                                 + 1.22576 w_k1_IBM_s017
                                 + 1.09889 w_k1_INTC_s017 + 1.1456 w_k1_KO_s017
                                 + 1.66627 w_k1_MAT_s017
                                 + 1.06206 w_k1_MSFT_s017
                                 + 1.17271 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.687994 w_k1_ADM_s017
                                 + 0.798493 w_k1_AXP_s017
                                 + 0.72486 w_k1_DIS_s017
                                 + 0.80041 w_k1_FDX_s017
                                 + 1.08643 w_k1_IBM_s017
                                 + 0.690007 w_k1_INTC_s017
                                 + 0.791278 w_k1_KO_s017
                                 + 1.59474 w_k1_MAT_s017
                                 + 0.787064 w_k1_MSFT_s017
                                 + 0.954668 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.949204 w_k1_ADM_s100
                                 + 0.677813 w_k1_AXP_s100
                                 + 0.635195 w_k1_DIS_s100
                                 + 0.797374 w_k1_FDX_s100
                                 + 0.869477 w_k1_IBM_s100
                                 + 1.03683 w_k1_INTC_s100
                                 + 0.746315 w_k1_KO_s100
                                 + 0.808644 w_k1_MAT_s100
                                 + 0.815601 w_k1_MSFT_s100
                                 + 0.836259 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.35381 w_k1_ADM_s100
                                 + 1.39023 w_k1_AXP_s100
                                 + 1.73699 w_k1_DIS_s100
                                 + 1.36868 w_k1_FDX_s100
                                 + 1.17835 w_k1_IBM_s100
                                 + 1.71826 w_k1_INTC_s100
                                 + 1.28628 w_k1_KO_s100 + 1.0492 w_k1_MAT_s100
                                 + 1.37869 w_k1_MSFT_s100
                                 + 1.17973 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.992644 w_k1_ADM_s100
                                 + 1.15791 w_k1_AXP_s100
                                 + 1.03103 w_k1_DIS_s100
                                 + 1.08732 w_k1_FDX_s100
                                 + 0.922846 w_k1_IBM_s100
                                 + 1.41148 w_k1_INTC_s100 + 1.1035 w_k1_KO_s100
                                 + 1.0118 w_k1_MAT_s100
                                 + 1.23055 w_k1_MSFT_s100
                                 + 1.11476 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.865385 w_k1_ADM_s100
                                 + 0.784018 w_k1_AXP_s100
                                 + 0.737692 w_k1_DIS_s100
                                 + 0.913628 w_k1_FDX_s100
                                 + 0.631938 w_k1_IBM_s100
                                 + 1.06854 w_k1_INTC_s100
                                 + 0.738004 w_k1_KO_s100
                                 + 0.547373 w_k1_MAT_s100
                                 + 0.888211 w_k1_MSFT_s100
                                 + 0.842179 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.46611 w_k1_ADM_s100
                                 + 1.85596 w_k1_AXP_s100
                                 + 1.35903 w_k1_DIS_s100 + 1.3863 w_k1_FDX_s100
                                 + 1.61234 w_k1_IBM_s100
                                 + 1.37632 w_k1_INTC_s100
                                 + 1.32601 w_k1_KO_s100 + 2.17024 w_k1_MAT_s100
                                 + 1.17271 w_k1_MSFT_s100
                                 + 1.26815 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.799365 w_k1_ADM_s100
                                 + 1.14016 w_k1_AXP_s100
                                 + 0.869696 w_k1_DIS_s100
                                 + 0.814281 w_k1_FDX_s100
                                 + 1.50366 w_k1_IBM_s100
                                 + 0.747391 w_k1_INTC_s100
                                 + 1.1173 w_k1_KO_s100 + 1.95285 w_k1_MAT_s100
                                 + 0.887287 w_k1_MSFT_s100
                                 + 1.06233 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.14653 w_k1_ADM_s100
                                 + 1.24603 w_k1_AXP_s100 + 1.4439 w_k1_DIS_s100
                                 + 1.13132 w_k1_FDX_s100
                                 + 1.29127 w_k1_IBM_s100
                                 + 1.4432 w_k1_INTC_s100 + 1.37648 w_k1_KO_s100
                                 + 1.56067 w_k1_MAT_s100 + 1.318 w_k1_MSFT_s100
                                 + 1.19087 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.08945 w_k1_ADM_s100
                                 + 0.874853 w_k1_AXP_s100
                                 + 0.899197 w_k1_DIS_s100
                                 + 0.859114 w_k1_FDX_s100
                                 + 1.12617 w_k1_IBM_s100
                                 + 1.20964 w_k1_INTC_s100
                                 + 1.11744 w_k1_KO_s100
                                 + 0.999109 w_k1_MAT_s100
                                 + 1.00795 w_k1_MSFT_s100 + 1.0276 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.01234 w_k1_ADM_s101
                                 + 0.999627 w_k1_AXP_s101
                                 + 1.05129 w_k1_DIS_s101
                                 + 0.991323 w_k1_FDX_s101
                                 + 0.857468 w_k1_IBM_s101
                                 + 0.933944 w_k1_INTC_s101
                                 + 1.01777 w_k1_KO_s101
                                 + 0.848571 w_k1_MAT_s101
                                 + 0.987597 w_k1_MSFT_s101
                                 + 0.96136 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.732458 w_k1_ADM_s101
                                 + 1.40335 w_k1_AXP_s101
                                 + 1.40924 w_k1_DIS_s101
                                 + 1.07457 w_k1_FDX_s101 + 1.7206 w_k1_IBM_s101
                                 + 1.67807 w_k1_INTC_s101
                                 + 1.33058 w_k1_KO_s101 + 1.60138 w_k1_MAT_s101
                                 + 1.2639 w_k1_MSFT_s101 + 1.13018 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.738342 w_k1_ADM_s101
                                 + 0.839202 w_k1_AXP_s101
                                 + 0.625601 w_k1_DIS_s101
                                 + 0.830023 w_k1_FDX_s101
                                 + 0.650797 w_k1_IBM_s101
                                 + 0.813504 w_k1_INTC_s101
                                 + 0.653979 w_k1_KO_s101
                                 + 0.632855 w_k1_MAT_s101
                                 + 0.852507 w_k1_MSFT_s101
                                 + 0.876555 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.650625 w_k1_ADM_s101
                                 + 0.726713 w_k1_AXP_s101
                                 + 0.773631 w_k1_DIS_s101
                                 + 0.790141 w_k1_FDX_s101
                                 + 0.786442 w_k1_IBM_s101
                                 + 0.869132 w_k1_INTC_s101
                                 + 0.728013 w_k1_KO_s101
                                 + 0.662426 w_k1_MAT_s101
                                 + 0.809409 w_k1_MSFT_s101
                                 + 0.786544 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.31198 w_k1_ADM_s101
                                 + 1.41787 w_k1_AXP_s101 + 1.2888 w_k1_DIS_s101
                                 + 1.25035 w_k1_FDX_s101
                                 + 0.90574 w_k1_IBM_s101
                                 + 1.2355 w_k1_INTC_s101 + 1.14152 w_k1_KO_s101
                                 + 0.951602 w_k1_MAT_s101
                                 + 1.26743 w_k1_MSFT_s101 + 1.2501 w_k1_PG_s101
                                 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.983387 w_k1_ADM_s101
                                 + 1.06763 w_k1_AXP_s101
                                 + 1.00714 w_k1_DIS_s101
                                 + 1.02728 w_k1_FDX_s101
                                 + 0.990514 w_k1_IBM_s101
                                 + 1.08875 w_k1_INTC_s101
                                 + 0.941089 w_k1_KO_s101
                                 + 0.843467 w_k1_MAT_s101
                                 + 1.01953 w_k1_MSFT_s101 + 1.0035 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.12347 w_k1_ADM_s101
                                 + 0.871662 w_k1_AXP_s101
                                 + 1.29276 w_k1_DIS_s101
                                 + 1.04033 w_k1_FDX_s101
                                 + 1.36529 w_k1_IBM_s101
                                 + 1.11894 w_k1_INTC_s101
                                 + 1.17777 w_k1_KO_s101 + 1.0726 w_k1_MAT_s101
                                 + 1.15173 w_k1_MSFT_s101
                                 + 1.02956 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.852123 w_k1_ADM_s101
                                 + 1.10988 w_k1_AXP_s101
                                 + 0.884577 w_k1_DIS_s101
                                 + 0.880822 w_k1_FDX_s101
                                 + 1.17301 w_k1_IBM_s101
                                 + 0.868036 w_k1_INTC_s101
                                 + 0.996418 w_k1_KO_s101
                                 + 1.39577 w_k1_MAT_s101
                                 + 0.897099 w_k1_MSFT_s101
                                 + 0.959889 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.03244 w_k1_ADM_s102
                                 + 0.866193 w_k1_AXP_s102
                                 + 0.882617 w_k1_DIS_s102
                                 + 0.912956 w_k1_FDX_s102
                                 + 0.905831 w_k1_IBM_s102
                                 + 1.03151 w_k1_INTC_s102
                                 + 1.16924 w_k1_KO_s102
                                 + 0.642765 w_k1_MAT_s102
                                 + 0.944201 w_k1_MSFT_s102
                                 + 1.01039 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.960435 w_k1_ADM_s102
                                 + 1.06194 w_k1_AXP_s102
                                 + 1.05781 w_k1_DIS_s102
                                 + 1.14446 w_k1_FDX_s102
                                 + 1.07007 w_k1_IBM_s102
                                 + 1.25139 w_k1_INTC_s102
                                 + 1.24564 w_k1_KO_s102 + 0.80056 w_k1_MAT_s102
                                 + 1.05619 w_k1_MSFT_s102
                                 + 0.937492 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.0739 w_k1_ADM_s102
                                 + 1.40099 w_k1_AXP_s102
                                 + 1.55679 w_k1_DIS_s102
                                 + 1.37672 w_k1_FDX_s102
                                 + 1.20512 w_k1_IBM_s102
                                 + 1.36783 w_k1_INTC_s102 + 1.1764 w_k1_KO_s102
                                 + 0.931351 w_k1_MAT_s102
                                 + 1.45198 w_k1_MSFT_s102
                                 + 1.09371 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.862472 w_k1_ADM_s102
                                 + 0.997085 w_k1_AXP_s102
                                 + 0.770939 w_k1_DIS_s102
                                 + 0.948303 w_k1_FDX_s102
                                 + 0.708361 w_k1_IBM_s102
                                 + 0.835748 w_k1_INTC_s102
                                 + 0.863227 w_k1_KO_s102
                                 + 1.01595 w_k1_MAT_s102
                                 + 0.892355 w_k1_MSFT_s102
                                 + 0.948878 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.22457 w_k1_ADM_s102
                                 + 1.47706 w_k1_AXP_s102
                                 + 1.39428 w_k1_DIS_s102 + 1.0622 w_k1_FDX_s102
                                 + 1.50673 w_k1_IBM_s102
                                 + 0.939277 w_k1_INTC_s102
                                 + 1.22642 w_k1_KO_s102 + 1.10844 w_k1_MAT_s102
                                 + 0.971265 w_k1_MSFT_s102
                                 + 1.16828 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.849004 w_k1_ADM_s102
                                 + 0.994588 w_k1_AXP_s102
                                 + 0.822686 w_k1_DIS_s102
                                 + 0.80972 w_k1_FDX_s102
                                 + 1.17124 w_k1_IBM_s102
                                 + 0.79938 w_k1_INTC_s102 + 1.149 w_k1_KO_s102
                                 + 1.29361 w_k1_MAT_s102
                                 + 0.829958 w_k1_MSFT_s102
                                 + 0.985485 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.615947 w_k1_ADM_s102
                                 + 0.696777 w_k1_AXP_s102
                                 + 0.580452 w_k1_DIS_s102
                                 + 0.745174 w_k1_FDX_s102
                                 + 0.813025 w_k1_IBM_s102
                                 + 0.882203 w_k1_INTC_s102
                                 + 0.892161 w_k1_KO_s102
                                 + 0.779063 w_k1_MAT_s102
                                 + 0.766796 w_k1_MSFT_s102
                                 + 0.817238 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.57013 w_k1_ADM_s102
                                 + 1.13077 w_k1_AXP_s102
                                 + 0.961461 w_k1_DIS_s102
                                 + 1.09504 w_k1_FDX_s102
                                 + 0.738807 w_k1_IBM_s102
                                 + 0.883484 w_k1_INTC_s102
                                 + 1.02876 w_k1_KO_s102
                                 + 0.610192 w_k1_MAT_s102
                                 + 0.902041 w_k1_MSFT_s102
                                 + 0.944341 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.36785 w_k1_ADM_s103
                                 + 1.09036 w_k1_AXP_s103
                                 + 1.10433 w_k1_DIS_s103
                                 + 1.17099 w_k1_FDX_s103
                                 + 0.829117 w_k1_IBM_s103
                                 + 1.07169 w_k1_INTC_s103
                                 + 0.960982 w_k1_KO_s103
                                 + 0.966375 w_k1_MAT_s103
                                 + 1.09193 w_k1_MSFT_s103
                                 + 1.04116 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.923606 w_k1_ADM_s103
                                 + 0.914226 w_k1_AXP_s103
                                 + 1.1643 w_k1_DIS_s103 + 1.02563 w_k1_FDX_s103
                                 + 1.07227 w_k1_IBM_s103
                                 + 1.23194 w_k1_INTC_s103
                                 + 1.17807 w_k1_KO_s103 + 1.12052 w_k1_MAT_s103
                                 + 1.27222 w_k1_MSFT_s103
                                 + 1.10137 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.948791 w_k1_ADM_s103
                                 + 0.72725 w_k1_AXP_s103
                                 + 0.689516 w_k1_DIS_s103
                                 + 0.805796 w_k1_FDX_s103
                                 + 0.669191 w_k1_IBM_s103
                                 + 0.897201 w_k1_INTC_s103
                                 + 0.686281 w_k1_KO_s103
                                 + 0.619418 w_k1_MAT_s103
                                 + 0.824178 w_k1_MSFT_s103
                                 + 0.875657 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.10196 w_k1_ADM_s103
                                 + 1.00572 w_k1_AXP_s103
                                 + 1.00263 w_k1_DIS_s103
                                 + 0.997481 w_k1_FDX_s103
                                 + 1.03612 w_k1_IBM_s103
                                 + 0.909692 w_k1_INTC_s103
                                 + 0.93598 w_k1_KO_s103 + 1.36552 w_k1_MAT_s103
                                 + 0.910192 w_k1_MSFT_s103
                                 + 0.979408 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.52362 w_k1_ADM_s103
                                 + 1.39454 w_k1_AXP_s103
                                 + 1.51416 w_k1_DIS_s103
                                 + 1.43418 w_k1_FDX_s103
                                 + 0.954263 w_k1_IBM_s103
                                 + 1.11881 w_k1_INTC_s103
                                 + 1.00394 w_k1_KO_s103 + 1.14906 w_k1_MAT_s103
                                 + 1.20731 w_k1_MSFT_s103
                                 + 1.10834 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.21156 w_k1_ADM_s103
                                 + 1.35471 w_k1_AXP_s103
                                 + 1.28578 w_k1_DIS_s103
                                 + 1.12799 w_k1_FDX_s103
                                 + 1.49532 w_k1_IBM_s103
                                 + 0.978259 w_k1_INTC_s103
                                 + 1.30021 w_k1_KO_s103 + 1.88906 w_k1_MAT_s103
                                 + 1.19284 w_k1_MSFT_s103
                                 + 1.24017 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.92259 w_k1_ADM_s103
                                 + 0.92858 w_k1_AXP_s103
                                 + 0.865273 w_k1_DIS_s103
                                 + 0.856293 w_k1_FDX_s103
                                 + 1.11601 w_k1_IBM_s103
                                 + 0.800555 w_k1_INTC_s103
                                 + 0.997901 w_k1_KO_s103
                                 + 1.31641 w_k1_MAT_s103
                                 + 0.906243 w_k1_MSFT_s103
                                 + 0.962812 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.970697 w_k1_ADM_s103
                                 + 0.643033 w_k1_AXP_s103
                                 + 0.886615 w_k1_DIS_s103
                                 + 0.993637 w_k1_FDX_s103
                                 + 0.665366 w_k1_IBM_s103
                                 + 1.01231 w_k1_INTC_s103
                                 + 0.870774 w_k1_KO_s103
                                 + 0.508772 w_k1_MAT_s103
                                 + 1.01467 w_k1_MSFT_s103
                                 + 1.04604 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.38933 w_k1_ADM_s104
                                 + 1.03613 w_k1_AXP_s104
                                 + 0.932958 w_k1_DIS_s104
                                 + 1.02109 w_k1_FDX_s104
                                 + 0.85323 w_k1_IBM_s104
                                 + 0.864276 w_k1_INTC_s104
                                 + 0.850444 w_k1_KO_s104
                                 + 0.650502 w_k1_MAT_s104
                                 + 0.84056 w_k1_MSFT_s104
                                 + 1.04346 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.06306 w_k1_ADM_s104
                                 + 0.739461 w_k1_AXP_s104
                                 + 0.586564 w_k1_DIS_s104
                                 + 0.813068 w_k1_FDX_s104
                                 + 0.638034 w_k1_IBM_s104
                                 + 0.729069 w_k1_INTC_s104
                                 + 0.56001 w_k1_KO_s104
                                 + 0.562619 w_k1_MAT_s104
                                 + 0.785965 w_k1_MSFT_s104
                                 + 0.763311 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.44894 w_k1_ADM_s104
                                 + 1.40681 w_k1_AXP_s104
                                 + 1.62777 w_k1_DIS_s104
                                 + 1.38268 w_k1_FDX_s104
                                 + 1.15344 w_k1_IBM_s104
                                 + 1.19496 w_k1_INTC_s104
                                 + 1.32826 w_k1_KO_s104 + 1.78292 w_k1_MAT_s104
                                 + 1.40039 w_k1_MSFT_s104 + 1.3311 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.943745 w_k1_ADM_s104
                                 + 0.750045 w_k1_AXP_s104
                                 + 0.589676 w_k1_DIS_s104
                                 + 0.71259 w_k1_FDX_s104
                                 + 0.899358 w_k1_IBM_s104
                                 + 0.73616 w_k1_INTC_s104
                                 + 0.727665 w_k1_KO_s104
                                 + 0.745853 w_k1_MAT_s104
                                 + 0.783463 w_k1_MSFT_s104
                                 + 0.84866 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.05434 w_k1_ADM_s104
                                 + 0.86242 w_k1_AXP_s104
                                 + 0.877392 w_k1_DIS_s104
                                 + 0.999194 w_k1_FDX_s104
                                 + 0.701055 w_k1_IBM_s104
                                 + 0.917281 w_k1_INTC_s104
                                 + 0.608408 w_k1_KO_s104
                                 + 0.500914 w_k1_MAT_s104
                                 + 0.97412 w_k1_MSFT_s104
                                 + 0.872683 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.00704 w_k1_ADM_s104
                                 + 0.728211 w_k1_AXP_s104
                                 + 0.832825 w_k1_DIS_s104
                                 + 0.781178 w_k1_FDX_s104
                                 + 0.990427 w_k1_IBM_s104
                                 + 0.933076 w_k1_INTC_s104
                                 + 0.795198 w_k1_KO_s104
                                 + 0.529809 w_k1_MAT_s104
                                 + 0.863088 w_k1_MSFT_s104
                                 + 0.829015 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.09398 w_k1_ADM_s104
                                 + 0.902491 w_k1_AXP_s104
                                 + 0.984205 w_k1_DIS_s104
                                 + 0.919514 w_k1_FDX_s104
                                 + 1.47184 w_k1_IBM_s104
                                 + 1.18001 w_k1_INTC_s104
                                 + 1.12784 w_k1_KO_s104
                                 + 0.820261 w_k1_MAT_s104
                                 + 1.00411 w_k1_MSFT_s104
                                 + 1.11708 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.23369 w_k1_ADM_s104
                                 + 0.984715 w_k1_AXP_s104
                                 + 1.18274 w_k1_DIS_s104
                                 + 1.08615 w_k1_FDX_s104
                                 + 1.01422 w_k1_IBM_s104
                                 + 0.980423 w_k1_INTC_s104
                                 + 0.996576 w_k1_KO_s104
                                 + 0.872319 w_k1_MAT_s104
                                 + 0.997203 w_k1_MSFT_s104
                                 + 1.12295 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.849037 w_k1_ADM_s105
                                 + 0.767293 w_k1_AXP_s105
                                 + 0.80099 w_k1_DIS_s105
                                 + 0.824222 w_k1_FDX_s105
                                 + 0.94103 w_k1_IBM_s105
                                 + 1.01385 w_k1_INTC_s105
                                 + 0.926481 w_k1_KO_s105
                                 + 1.11704 w_k1_MAT_s105
                                 + 1.08608 w_k1_MSFT_s105
                                 + 0.908395 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.809869 w_k1_ADM_s105
                                 + 0.963983 w_k1_AXP_s105
                                 + 0.724707 w_k1_DIS_s105
                                 + 0.793648 w_k1_FDX_s105
                                 + 1.27727 w_k1_IBM_s105
                                 + 0.874329 w_k1_INTC_s105
                                 + 0.896415 w_k1_KO_s105
                                 + 1.81165 w_k1_MAT_s105
                                 + 0.794667 w_k1_MSFT_s105
                                 + 0.86524 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.07685 w_k1_ADM_s105
                                 + 1.00219 w_k1_AXP_s105
                                 + 1.19207 w_k1_DIS_s105 + 1.0239 w_k1_FDX_s105
                                 + 1.42585 w_k1_IBM_s105
                                 + 0.90853 w_k1_INTC_s105 + 1.0756 w_k1_KO_s105
                                 + 1.79956 w_k1_MAT_s105
                                 + 1.05849 w_k1_MSFT_s105
                                 + 1.02698 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.37544 w_k1_ADM_s105
                                 + 1.66581 w_k1_AXP_s105
                                 + 1.84155 w_k1_DIS_s105
                                 + 1.41096 w_k1_FDX_s105
                                 + 1.40414 w_k1_IBM_s105
                                 + 1.66491 w_k1_INTC_s105
                                 + 1.52632 w_k1_KO_s105 + 2.24488 w_k1_MAT_s105
                                 + 1.55426 w_k1_MSFT_s105
                                 + 1.28776 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.36331 w_k1_ADM_s105
                                 + 1.37643 w_k1_AXP_s105 + 1.4629 w_k1_DIS_s105
                                 + 1.20377 w_k1_FDX_s105
                                 + 1.34675 w_k1_IBM_s105
                                 + 1.4487 w_k1_INTC_s105 + 1.31073 w_k1_KO_s105
                                 + 1.69117 w_k1_MAT_s105
                                 + 1.29971 w_k1_MSFT_s105
                                 + 1.12494 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.942012 w_k1_ADM_s105
                                 + 0.60173 w_k1_AXP_s105
                                 + 0.574823 w_k1_DIS_s105
                                 + 0.771834 w_k1_FDX_s105
                                 + 0.584721 w_k1_IBM_s105
                                 + 0.833564 w_k1_INTC_s105
                                 + 0.633655 w_k1_KO_s105
                                 + 0.478299 w_k1_MAT_s105
                                 + 0.798658 w_k1_MSFT_s105
                                 + 0.726703 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.888364 w_k1_ADM_s105
                                 + 0.959742 w_k1_AXP_s105
                                 + 0.82889 w_k1_DIS_s105
                                 + 1.03435 w_k1_FDX_s105
                                 + 0.644906 w_k1_IBM_s105
                                 + 1.23695 w_k1_INTC_s105
                                 + 0.769585 w_k1_KO_s105
                                 + 1.09074 w_k1_MAT_s105
                                 + 1.09092 w_k1_MSFT_s105
                                 + 0.874348 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.00203 w_k1_ADM_s105
                                 + 1.11202 w_k1_AXP_s105
                                 + 1.14426 w_k1_DIS_s105
                                 + 1.18604 w_k1_FDX_s105
                                 + 0.889796 w_k1_IBM_s105
                                 + 1.18002 w_k1_INTC_s105
                                 + 1.03642 w_k1_KO_s105 + 1.22707 w_k1_MAT_s105
                                 + 1.11318 w_k1_MSFT_s105 + 1.056 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.947513 w_k1_ADM_s106
                                 + 0.92551 w_k1_AXP_s106
                                 + 1.07014 w_k1_DIS_s106
                                 + 0.875792 w_k1_FDX_s106
                                 + 1.36247 w_k1_IBM_s106
                                 + 0.971823 w_k1_INTC_s106
                                 + 1.16279 w_k1_KO_s106 + 1.05121 w_k1_MAT_s106
                                 + 0.985809 w_k1_MSFT_s106
                                 + 0.974487 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.44272 w_k1_ADM_s106
                                 + 1.24389 w_k1_AXP_s106
                                 + 1.61289 w_k1_DIS_s106
                                 + 1.33899 w_k1_FDX_s106
                                 + 1.17032 w_k1_IBM_s106
                                 + 1.14483 w_k1_INTC_s106
                                 + 1.32222 w_k1_KO_s106 + 1.17159 w_k1_MAT_s106
                                 + 1.33184 w_k1_MSFT_s106
                                 + 1.12044 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.990734 w_k1_ADM_s106
                                 + 0.662982 w_k1_AXP_s106
                                 + 0.84853 w_k1_DIS_s106
                                 + 0.924695 w_k1_FDX_s106
                                 + 0.718165 w_k1_IBM_s106
                                 + 0.949965 w_k1_INTC_s106
                                 + 0.883013 w_k1_KO_s106
                                 + 0.427737 w_k1_MAT_s106
                                 + 0.955972 w_k1_MSFT_s106
                                 + 0.861371 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.34408 w_k1_ADM_s106
                                 + 1.43914 w_k1_AXP_s106
                                 + 1.19103 w_k1_DIS_s106
                                 + 1.13165 w_k1_FDX_s106 + 1.4814 w_k1_IBM_s106
                                 + 0.999084 w_k1_INTC_s106
                                 + 1.27114 w_k1_KO_s106 + 1.93541 w_k1_MAT_s106
                                 + 1.05609 w_k1_MSFT_s106 + 1.1219 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.1056 w_k1_ADM_s106
                                 + 1.25048 w_k1_AXP_s106
                                 + 1.19576 w_k1_DIS_s106 + 1.3122 w_k1_FDX_s106
                                 + 0.889078 w_k1_IBM_s106
                                 + 0.927009 w_k1_INTC_s106
                                 + 1.06719 w_k1_KO_s106 + 1.0075 w_k1_MAT_s106
                                 + 1.07317 w_k1_MSFT_s106
                                 + 1.07518 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.27007 w_k1_ADM_s106
                                 + 0.952073 w_k1_AXP_s106
                                 + 1.22626 w_k1_DIS_s106
                                 + 1.04784 w_k1_FDX_s106
                                 + 1.10024 w_k1_IBM_s106
                                 + 0.940494 w_k1_INTC_s106
                                 + 1.14485 w_k1_KO_s106
                                 + 0.732545 w_k1_MAT_s106
                                 + 1.18172 w_k1_MSFT_s106
                                 + 1.08231 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.657534 w_k1_ADM_s106
                                 + 0.730638 w_k1_AXP_s106
                                 + 0.743435 w_k1_DIS_s106
                                 + 0.770675 w_k1_FDX_s106
                                 + 0.879487 w_k1_IBM_s106
                                 + 0.82862 w_k1_INTC_s106
                                 + 0.879499 w_k1_KO_s106
                                 + 0.842486 w_k1_MAT_s106
                                 + 0.944219 w_k1_MSFT_s106
                                 + 0.912916 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.883442 w_k1_ADM_s106
                                 + 0.920624 w_k1_AXP_s106
                                 + 0.846967 w_k1_DIS_s106
                                 + 0.92322 w_k1_FDX_s106 + 1.0618 w_k1_IBM_s106
                                 + 0.918338 w_k1_INTC_s106
                                 + 1.01855 w_k1_KO_s106
                                 + 0.915301 w_k1_MAT_s106
                                 + 0.889646 w_k1_MSFT_s106
                                 + 0.885571 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.889736 w_k1_ADM_s107
                                 + 0.852593 w_k1_AXP_s107
                                 + 0.873987 w_k1_DIS_s107
                                 + 0.919258 w_k1_FDX_s107
                                 + 0.903531 w_k1_IBM_s107
                                 + 1.00796 w_k1_INTC_s107
                                 + 0.866375 w_k1_KO_s107
                                 + 0.691464 w_k1_MAT_s107
                                 + 0.883225 w_k1_MSFT_s107
                                 + 0.917705 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.757193 w_k1_ADM_s107
                                 + 0.850864 w_k1_AXP_s107
                                 + 0.844206 w_k1_DIS_s107
                                 + 0.811774 w_k1_FDX_s107
                                 + 1.21418 w_k1_IBM_s107
                                 + 0.791516 w_k1_INTC_s107
                                 + 1.12771 w_k1_KO_s107 + 1.52244 w_k1_MAT_s107
                                 + 0.847386 w_k1_MSFT_s107
                                 + 0.963178 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.786254 w_k1_ADM_s107
                                 + 0.65655 w_k1_AXP_s107
                                 + 0.58564 w_k1_DIS_s107
                                 + 0.703155 w_k1_FDX_s107
                                 + 0.802656 w_k1_IBM_s107
                                 + 0.793325 w_k1_INTC_s107
                                 + 0.726966 w_k1_KO_s107
                                 + 0.880035 w_k1_MAT_s107
                                 + 0.735236 w_k1_MSFT_s107
                                 + 0.807558 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.867339 w_k1_ADM_s107
                                 + 1.08106 w_k1_AXP_s107
                                 + 1.52677 w_k1_DIS_s107
                                 + 1.14025 w_k1_FDX_s107
                                 + 1.48287 w_k1_IBM_s107
                                 + 1.07089 w_k1_INTC_s107
                                 + 1.37592 w_k1_KO_s107 + 1.74134 w_k1_MAT_s107
                                 + 1.22388 w_k1_MSFT_s107 + 1.4199 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.07586 w_k1_ADM_s107
                                 + 0.978197 w_k1_AXP_s107
                                 + 1.24317 w_k1_DIS_s107
                                 + 1.16724 w_k1_FDX_s107
                                 + 0.689631 w_k1_IBM_s107
                                 + 1.11815 w_k1_INTC_s107
                                 + 0.896188 w_k1_KO_s107
                                 + 0.596531 w_k1_MAT_s107
                                 + 1.0577 w_k1_MSFT_s107
                                 + 0.937652 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.745541 w_k1_ADM_s107
                                 + 0.579157 w_k1_AXP_s107
                                 + 0.743574 w_k1_DIS_s107
                                 + 0.882394 w_k1_FDX_s107
                                 + 0.614522 w_k1_IBM_s107
                                 + 0.916482 w_k1_INTC_s107
                                 + 0.811699 w_k1_KO_s107
                                 + 0.591079 w_k1_MAT_s107
                                 + 0.901033 w_k1_MSFT_s107
                                 + 0.893734 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.38306 w_k1_ADM_s107
                                 + 1.60033 w_k1_AXP_s107
                                 + 2.05618 w_k1_DIS_s107
                                 + 1.54929 w_k1_FDX_s107
                                 + 1.23897 w_k1_IBM_s107
                                 + 1.47623 w_k1_INTC_s107
                                 + 1.50126 w_k1_KO_s107 + 1.52809 w_k1_MAT_s107
                                 + 1.43652 w_k1_MSFT_s107 + 1.2935 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.32913 w_k1_ADM_s107
                                 + 1.25726 w_k1_AXP_s107
                                 + 1.04994 w_k1_DIS_s107
                                 + 1.16795 w_k1_FDX_s107
                                 + 1.19403 w_k1_IBM_s107
                                 + 1.12912 w_k1_INTC_s107
                                 + 1.11383 w_k1_KO_s107 + 1.49527 w_k1_MAT_s107
                                 + 1.08448 w_k1_MSFT_s107
                                 + 1.12953 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.932131 w_k1_ADM_s110
                                 + 0.826929 w_k1_AXP_s110
                                 + 1.10173 w_k1_DIS_s110
                                 + 0.892154 w_k1_FDX_s110
                                 + 1.41911 w_k1_IBM_s110
                                 + 0.866685 w_k1_INTC_s110
                                 + 1.21587 w_k1_KO_s110 + 1.46291 w_k1_MAT_s110
                                 + 0.958161 w_k1_MSFT_s110
                                 + 1.03243 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.821082 w_k1_ADM_s110
                                 + 0.705323 w_k1_AXP_s110
                                 + 0.571143 w_k1_DIS_s110
                                 + 0.730232 w_k1_FDX_s110
                                 + 0.832074 w_k1_IBM_s110
                                 + 0.782932 w_k1_INTC_s110
                                 + 0.763327 w_k1_KO_s110
                                 + 0.683058 w_k1_MAT_s110
                                 + 0.758112 w_k1_MSFT_s110
                                 + 0.865449 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.36568 w_k1_ADM_s110
                                 + 1.42214 w_k1_AXP_s110
                                 + 1.72997 w_k1_DIS_s110
                                 + 1.46323 w_k1_FDX_s110
                                 + 1.22151 w_k1_IBM_s110
                                 + 1.19525 w_k1_INTC_s110
                                 + 1.39163 w_k1_KO_s110 + 1.47645 w_k1_MAT_s110
                                 + 1.28128 w_k1_MSFT_s110 + 1.3169 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.961044 w_k1_ADM_s110
                                 + 0.839214 w_k1_AXP_s110
                                 + 0.717167 w_k1_DIS_s110
                                 + 0.905159 w_k1_FDX_s110
                                 + 0.715628 w_k1_IBM_s110
                                 + 0.89094 w_k1_INTC_s110
                                 + 0.811954 w_k1_KO_s110
                                 + 0.54074 w_k1_MAT_s110
                                 + 0.85952 w_k1_MSFT_s110
                                 + 0.825474 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.16311 w_k1_ADM_s110
                                 + 1.30645 w_k1_AXP_s110
                                 + 1.35483 w_k1_DIS_s110
                                 + 1.28655 w_k1_FDX_s110
                                 + 1.03151 w_k1_IBM_s110
                                 + 1.18718 w_k1_INTC_s110
                                 + 1.18544 w_k1_KO_s110 + 1.16226 w_k1_MAT_s110
                                 + 1.19074 w_k1_MSFT_s110
                                 + 1.06841 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.05614 w_k1_ADM_s110
                                 + 1.50626 w_k1_AXP_s110
                                 + 1.24965 w_k1_DIS_s110
                                 + 1.25326 w_k1_FDX_s110
                                 + 1.53277 w_k1_IBM_s110
                                 + 1.18806 w_k1_INTC_s110
                                 + 1.36324 w_k1_KO_s110 + 2.17776 w_k1_MAT_s110
                                 + 1.16161 w_k1_MSFT_s110
                                 + 1.12174 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.00633 w_k1_ADM_s110
                                 + 0.927941 w_k1_AXP_s110
                                 + 1.22461 w_k1_DIS_s110
                                 + 1.10746 w_k1_FDX_s110 + 1.0162 w_k1_IBM_s110
                                 + 1.11159 w_k1_INTC_s110
                                 + 1.02855 w_k1_KO_s110 + 0.97837 w_k1_MAT_s110
                                 + 1.16779 w_k1_MSFT_s110
                                 + 1.04958 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.957159 w_k1_ADM_s110
                                 + 0.952396 w_k1_AXP_s110
                                 + 0.926929 w_k1_DIS_s110
                                 + 0.943638 w_k1_FDX_s110
                                 + 1.09982 w_k1_IBM_s110
                                 + 0.911045 w_k1_INTC_s110
                                 + 1.11636 w_k1_KO_s110 + 1.00634 w_k1_MAT_s110
                                 + 0.916665 w_k1_MSFT_s110
                                 + 0.995124 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.677523 w_k1_ADM_s111
                                 + 0.577216 w_k1_AXP_s111
                                 + 0.628936 w_k1_DIS_s111
                                 + 0.776919 w_k1_FDX_s111
                                 + 0.612258 w_k1_IBM_s111
                                 + 0.899628 w_k1_INTC_s111
                                 + 0.700358 w_k1_KO_s111
                                 + 0.561742 w_k1_MAT_s111
                                 + 0.863101 w_k1_MSFT_s111
                                 + 0.817137 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.06636 w_k1_ADM_s111
                                 + 1.26846 w_k1_AXP_s111
                                 + 1.44129 w_k1_DIS_s111
                                 + 1.39733 w_k1_FDX_s111
                                 + 0.929716 w_k1_IBM_s111
                                 + 1.43926 w_k1_INTC_s111
                                 + 1.34287 w_k1_KO_s111 + 1.11798 w_k1_MAT_s111
                                 + 1.47249 w_k1_MSFT_s111
                                 + 1.23356 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.949459 w_k1_ADM_s111
                                 + 0.838267 w_k1_AXP_s111
                                 + 0.948117 w_k1_DIS_s111
                                 + 0.907883 w_k1_FDX_s111
                                 + 1.15735 w_k1_IBM_s111
                                 + 1.05528 w_k1_INTC_s111
                                 + 1.13571 w_k1_KO_s111
                                 + 0.968097 w_k1_MAT_s111
                                 + 0.989578 w_k1_MSFT_s111
                                 + 0.985848 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.1064 w_k1_ADM_s111
                                 + 0.87078 w_k1_AXP_s111
                                 + 1.22948 w_k1_DIS_s111 + 1.1329 w_k1_FDX_s111
                                 + 1.1236 w_k1_IBM_s111
                                 + 1.03666 w_k1_INTC_s111 + 1.1396 w_k1_KO_s111
                                 + 0.800527 w_k1_MAT_s111
                                 + 1.16304 w_k1_MSFT_s111
                                 + 1.11373 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.87484 w_k1_ADM_s111
                                 + 0.926638 w_k1_AXP_s111
                                 + 0.9673 w_k1_DIS_s111 + 1.04651 w_k1_FDX_s111
                                 + 0.775685 w_k1_IBM_s111
                                 + 1.04695 w_k1_INTC_s111
                                 + 0.832718 w_k1_KO_s111
                                 + 1.02291 w_k1_MAT_s111
                                 + 1.00948 w_k1_MSFT_s111
                                 + 0.918127 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.05271 w_k1_ADM_s111
                                 + 1.16144 w_k1_AXP_s111
                                 + 0.999102 w_k1_DIS_s111
                                 + 1.10339 w_k1_FDX_s111
                                 + 0.936672 w_k1_IBM_s111
                                 + 0.990873 w_k1_INTC_s111
                                 + 1.02369 w_k1_KO_s111 + 1.22981 w_k1_MAT_s111
                                 + 1.03215 w_k1_MSFT_s111
                                 + 1.04573 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.15628 w_k1_ADM_s111
                                 + 1.58758 w_k1_AXP_s111
                                 + 1.35502 w_k1_DIS_s111
                                 + 1.30532 w_k1_FDX_s111
                                 + 1.47709 w_k1_IBM_s111
                                 + 1.30178 w_k1_INTC_s111
                                 + 1.45523 w_k1_KO_s111 + 1.59642 w_k1_MAT_s111
                                 + 1.29032 w_k1_MSFT_s111
                                 + 1.24066 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.832236 w_k1_ADM_s111
                                 + 0.866349 w_k1_AXP_s111
                                 + 0.715753 w_k1_DIS_s111
                                 + 0.851085 w_k1_FDX_s111
                                 + 0.977451 w_k1_IBM_s111
                                 + 0.887883 w_k1_INTC_s111
                                 + 0.990748 w_k1_KO_s111
                                 + 1.01706 w_k1_MAT_s111
                                 + 0.888481 w_k1_MSFT_s111
                                 + 0.930817 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.01191 w_k1_ADM_s112
                                 + 1.09457 w_k1_AXP_s112 + 1.3402 w_k1_DIS_s112
                                 + 1.10072 w_k1_FDX_s112
                                 + 1.35861 w_k1_IBM_s112
                                 + 1.23641 w_k1_INTC_s112
                                 + 1.38983 w_k1_KO_s112 + 1.57876 w_k1_MAT_s112
                                 + 1.13432 w_k1_MSFT_s112
                                 + 1.01746 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.03388 w_k1_ADM_s112
                                 + 0.957137 w_k1_AXP_s112
                                 + 1.16229 w_k1_DIS_s112
                                 + 0.916041 w_k1_FDX_s112
                                 + 1.77121 w_k1_IBM_s112
                                 + 1.03856 w_k1_INTC_s112
                                 + 1.36172 w_k1_KO_s112 + 1.3986 w_k1_MAT_s112
                                 + 1.00047 w_k1_MSFT_s112
                                 + 1.10683 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.21658 w_k1_ADM_s112
                                 + 1.43532 w_k1_AXP_s112
                                 + 1.71251 w_k1_DIS_s112
                                 + 1.32939 w_k1_FDX_s112
                                 + 1.24742 w_k1_IBM_s112
                                 + 1.4421 w_k1_INTC_s112 + 1.21807 w_k1_KO_s112
                                 + 1.17427 w_k1_MAT_s112
                                 + 1.25993 w_k1_MSFT_s112
                                 + 1.10513 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.879472 w_k1_ADM_s112
                                 + 1.28643 w_k1_AXP_s112
                                 + 0.95255 w_k1_DIS_s112
                                 + 0.970568 w_k1_FDX_s112
                                 + 1.01216 w_k1_IBM_s112
                                 + 1.08857 w_k1_INTC_s112
                                 + 1.02031 w_k1_KO_s112 + 1.32237 w_k1_MAT_s112
                                 + 0.908679 w_k1_MSFT_s112
                                 + 1.04978 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.982378 w_k1_ADM_s112
                                 + 0.981061 w_k1_AXP_s112
                                 + 1.22125 w_k1_DIS_s112
                                 + 1.07447 w_k1_FDX_s112
                                 + 0.94011 w_k1_IBM_s112
                                 + 1.13117 w_k1_INTC_s112
                                 + 1.03242 w_k1_KO_s112
                                 + 0.796137 w_k1_MAT_s112
                                 + 1.09217 w_k1_MSFT_s112
                                 + 0.979042 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.841787 w_k1_ADM_s112
                                 + 0.641282 w_k1_AXP_s112
                                 + 0.566139 w_k1_DIS_s112
                                 + 0.746947 w_k1_FDX_s112
                                 + 0.706121 w_k1_IBM_s112
                                 + 0.912744 w_k1_INTC_s112
                                 + 0.79843 w_k1_KO_s112
                                 + 0.740278 w_k1_MAT_s112
                                 + 0.795767 w_k1_MSFT_s112
                                 + 0.771218 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.852253 w_k1_ADM_s112
                                 + 0.83365 w_k1_AXP_s112
                                 + 0.776007 w_k1_DIS_s112
                                 + 0.956397 w_k1_FDX_s112
                                 + 0.778722 w_k1_IBM_s112
                                 + 1.12016 w_k1_INTC_s112
                                 + 0.909531 w_k1_KO_s112
                                 + 0.824068 w_k1_MAT_s112
                                 + 1.00476 w_k1_MSFT_s112
                                 + 0.968113 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.832639 w_k1_ADM_s112
                                 + 0.887478 w_k1_AXP_s112
                                 + 0.904685 w_k1_DIS_s112
                                 + 0.836284 w_k1_FDX_s112
                                 + 1.18728 w_k1_IBM_s112
                                 + 0.982271 w_k1_INTC_s112
                                 + 1.05232 w_k1_KO_s112 + 1.02129 w_k1_MAT_s112
                                 + 0.987805 w_k1_MSFT_s112
                                 + 0.870508 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.01051 w_k1_ADM_s113
                                 + 1.08303 w_k1_AXP_s113
                                 + 0.707476 w_k1_DIS_s113
                                 + 1.00819 w_k1_FDX_s113
                                 + 0.798032 w_k1_IBM_s113
                                 + 0.838155 w_k1_INTC_s113
                                 + 0.647453 w_k1_KO_s113
                                 + 0.665187 w_k1_MAT_s113
                                 + 0.730212 w_k1_MSFT_s113
                                 + 0.93464 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.14933 w_k1_ADM_s113
                                 + 0.948971 w_k1_AXP_s113
                                 + 1.12361 w_k1_DIS_s113
                                 + 1.16151 w_k1_FDX_s113
                                 + 0.79896 w_k1_IBM_s113
                                 + 0.878752 w_k1_INTC_s113
                                 + 0.848912 w_k1_KO_s113
                                 + 0.737778 w_k1_MAT_s113
                                 + 1.0035 w_k1_MSFT_s113 + 1.08111 w_k1_PG_s113
                                 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.06775 w_k1_ADM_s113
                                 + 1.13882 w_k1_AXP_s113
                                 + 1.30644 w_k1_DIS_s113
                                 + 1.16747 w_k1_FDX_s113
                                 + 0.97367 w_k1_IBM_s113
                                 + 1.15086 w_k1_INTC_s113
                                 + 1.01274 w_k1_KO_s113 + 0.81827 w_k1_MAT_s113
                                 + 1.23013 w_k1_MSFT_s113
                                 + 1.06461 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.880389 w_k1_ADM_s113
                                 + 1.03206 w_k1_AXP_s113
                                 + 0.878843 w_k1_DIS_s113
                                 + 0.897846 w_k1_FDX_s113
                                 + 1.35429 w_k1_IBM_s113
                                 + 0.749187 w_k1_INTC_s113
                                 + 0.995217 w_k1_KO_s113 + 1.2156 w_k1_MAT_s113
                                 + 0.781857 w_k1_MSFT_s113
                                 + 1.03243 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.14189 w_k1_ADM_s113
                                 + 1.49767 w_k1_AXP_s113
                                 + 0.983574 w_k1_DIS_s113
                                 + 1.02129 w_k1_FDX_s113
                                 + 1.21525 w_k1_IBM_s113
                                 + 0.960946 w_k1_INTC_s113
                                 + 0.89437 w_k1_KO_s113 + 1.44546 w_k1_MAT_s113
                                 + 0.838535 w_k1_MSFT_s113
                                 + 1.06153 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.798089 w_k1_ADM_s113
                                 + 0.677119 w_k1_AXP_s113
                                 + 0.788994 w_k1_DIS_s113
                                 + 0.807054 w_k1_FDX_s113
                                 + 0.807424 w_k1_IBM_s113
                                 + 0.782377 w_k1_INTC_s113
                                 + 0.770737 w_k1_KO_s113
                                 + 0.775568 w_k1_MAT_s113
                                 + 0.776188 w_k1_MSFT_s113
                                 + 0.920235 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.44175 w_k1_ADM_s113
                                 + 1.44928 w_k1_AXP_s113
                                 + 1.59901 w_k1_DIS_s113
                                 + 1.34457 w_k1_FDX_s113
                                 + 1.27446 w_k1_IBM_s113
                                 + 1.05646 w_k1_INTC_s113 + 1.041 w_k1_KO_s113
                                 + 1.4645 w_k1_MAT_s113
                                 + 1.20908 w_k1_MSFT_s113
                                 + 1.28753 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.958749 w_k1_ADM_s113
                                 + 0.893682 w_k1_AXP_s113
                                 + 0.882889 w_k1_DIS_s113
                                 + 1.04141 w_k1_FDX_s113
                                 + 0.974198 w_k1_IBM_s113
                                 + 0.937719 w_k1_INTC_s113
                                 + 0.789545 w_k1_KO_s113
                                 + 0.731088 w_k1_MAT_s113
                                 + 0.892247 w_k1_MSFT_s113
                                 + 1.04596 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.12985 w_k1_ADM_s114
                                 + 0.837556 w_k1_AXP_s114
                                 + 1.15911 w_k1_DIS_s114
                                 + 1.10005 w_k1_FDX_s114
                                 + 0.929633 w_k1_IBM_s114
                                 + 1.00833 w_k1_INTC_s114
                                 + 1.10329 w_k1_KO_s114 + 1.02391 w_k1_MAT_s114
                                 + 1.04165 w_k1_MSFT_s114
                                 + 1.21586 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.795709 w_k1_ADM_s114
                                 + 0.691461 w_k1_AXP_s114
                                 + 0.817219 w_k1_DIS_s114
                                 + 0.874723 w_k1_FDX_s114
                                 + 0.699497 w_k1_IBM_s114
                                 + 0.786364 w_k1_INTC_s114
                                 + 0.892405 w_k1_KO_s114
                                 + 0.597656 w_k1_MAT_s114
                                 + 0.84259 w_k1_MSFT_s114
                                 + 0.951947 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.877135 w_k1_ADM_s114
                                 + 0.755844 w_k1_AXP_s114
                                 + 0.656674 w_k1_DIS_s114
                                 + 0.788623 w_k1_FDX_s114
                                 + 1.05617 w_k1_IBM_s114
                                 + 0.843903 w_k1_INTC_s114
                                 + 0.915437 w_k1_KO_s114
                                 + 1.28182 w_k1_MAT_s114
                                 + 0.803444 w_k1_MSFT_s114
                                 + 1.07055 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.912216 w_k1_ADM_s114
                                 + 0.803242 w_k1_AXP_s114
                                 + 0.623924 w_k1_DIS_s114
                                 + 0.770435 w_k1_FDX_s114
                                 + 0.75477 w_k1_IBM_s114
                                 + 0.778513 w_k1_INTC_s114
                                 + 0.793172 w_k1_KO_s114
                                 + 0.810833 w_k1_MAT_s114
                                 + 0.770639 w_k1_MSFT_s114
                                 + 0.86311 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.818408 w_k1_ADM_s114
                                 + 0.980421 w_k1_AXP_s114
                                 + 0.982139 w_k1_DIS_s114
                                 + 0.90095 w_k1_FDX_s114
                                 + 1.55835 w_k1_IBM_s114
                                 + 0.892526 w_k1_INTC_s114
                                 + 1.0809 w_k1_KO_s114 + 1.65499 w_k1_MAT_s114
                                 + 0.872613 w_k1_MSFT_s114
                                 + 1.03839 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.971844 w_k1_ADM_s114
                                 + 1.17608 w_k1_AXP_s114
                                 + 0.976692 w_k1_DIS_s114
                                 + 1.07268 w_k1_FDX_s114 + 1.1468 w_k1_IBM_s114
                                 + 1.06344 w_k1_INTC_s114
                                 + 1.20202 w_k1_KO_s114 + 1.3883 w_k1_MAT_s114
                                 + 0.981496 w_k1_MSFT_s114
                                 + 1.19715 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.986811 w_k1_ADM_s114
                                 + 0.891899 w_k1_AXP_s114
                                 + 0.795549 w_k1_DIS_s114
                                 + 1.02097 w_k1_FDX_s114
                                 + 0.72065 w_k1_IBM_s114
                                 + 0.862382 w_k1_INTC_s114
                                 + 0.823146 w_k1_KO_s114
                                 + 0.970148 w_k1_MAT_s114
                                 + 0.873124 w_k1_MSFT_s114
                                 + 1.09982 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.5841 w_k1_ADM_s114
                                 + 1.37949 w_k1_AXP_s114
                                 + 1.67148 w_k1_DIS_s114
                                 + 1.46931 w_k1_FDX_s114
                                 + 0.972976 w_k1_IBM_s114
                                 + 1.14558 w_k1_INTC_s114
                                 + 1.23378 w_k1_KO_s114 + 1.00702 w_k1_MAT_s114
                                 + 1.2608 w_k1_MSFT_s114 + 1.33022 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.10564 w_k1_ADM_s115
                                 + 1.57704 w_k1_AXP_s115
                                 + 1.77181 w_k1_DIS_s115
                                 + 1.46866 w_k1_FDX_s115
                                 + 1.23092 w_k1_IBM_s115
                                 + 1.53451 w_k1_INTC_s115 + 1.2708 w_k1_KO_s115
                                 + 1.37667 w_k1_MAT_s115
                                 + 1.35575 w_k1_MSFT_s115 + 1.1796 w_k1_PG_s115
                                 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.794433 w_k1_ADM_s115
                                 + 0.925121 w_k1_AXP_s115
                                 + 0.849087 w_k1_DIS_s115
                                 + 0.852684 w_k1_FDX_s115
                                 + 1.04883 w_k1_IBM_s115
                                 + 0.98145 w_k1_INTC_s115
                                 + 1.11735 w_k1_KO_s115 + 1.323 w_k1_MAT_s115
                                 + 0.978301 w_k1_MSFT_s115
                                 + 0.941496 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.850319 w_k1_ADM_s115
                                 + 0.69851 w_k1_AXP_s115
                                 + 0.612014 w_k1_DIS_s115
                                 + 0.734525 w_k1_FDX_s115
                                 + 0.588027 w_k1_IBM_s115
                                 + 0.882521 w_k1_INTC_s115
                                 + 0.827704 w_k1_KO_s115
                                 + 0.792345 w_k1_MAT_s115
                                 + 0.834177 w_k1_MSFT_s115
                                 + 0.815113 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.942434 w_k1_ADM_s115
                                 + 1.17328 w_k1_AXP_s115
                                 + 1.34408 w_k1_DIS_s115
                                 + 0.986651 w_k1_FDX_s115
                                 + 1.83467 w_k1_IBM_s115
                                 + 1.17328 w_k1_INTC_s115
                                 + 1.67332 w_k1_KO_s115 + 2.4478 w_k1_MAT_s115
                                 + 1.18841 w_k1_MSFT_s115
                                 + 1.11958 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.37277 w_k1_ADM_s115
                                 + 1.25061 w_k1_AXP_s115
                                 + 1.46858 w_k1_DIS_s115
                                 + 1.38599 w_k1_FDX_s115
                                 + 0.820789 w_k1_IBM_s115
                                 + 1.03312 w_k1_INTC_s115
                                 + 1.12625 w_k1_KO_s115 + 1.15005 w_k1_MAT_s115
                                 + 1.43832 w_k1_MSFT_s115
                                 + 1.23048 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.945563 w_k1_ADM_s115
                                 + 0.840631 w_k1_AXP_s115
                                 + 0.957398 w_k1_DIS_s115
                                 + 1.0389 w_k1_FDX_s115
                                 + 0.607859 w_k1_IBM_s115
                                 + 1.14646 w_k1_INTC_s115
                                 + 0.976159 w_k1_KO_s115
                                 + 0.797692 w_k1_MAT_s115
                                 + 1.20125 w_k1_MSFT_s115
                                 + 0.923839 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.12264 w_k1_ADM_s115
                                 + 1.26564 w_k1_AXP_s115
                                 + 1.27058 w_k1_DIS_s115
                                 + 1.01238 w_k1_FDX_s115
                                 + 1.37533 w_k1_IBM_s115
                                 + 1.03003 w_k1_INTC_s115
                                 + 1.69767 w_k1_KO_s115 + 2.02972 w_k1_MAT_s115
                                 + 1.1808 w_k1_MSFT_s115 + 1.10272 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.11071 w_k1_ADM_s115
                                 + 1.16599 w_k1_AXP_s115
                                 + 1.23854 w_k1_DIS_s115
                                 + 1.14609 w_k1_FDX_s115
                                 + 0.960485 w_k1_IBM_s115
                                 + 1.04793 w_k1_INTC_s115
                                 + 1.16365 w_k1_KO_s115
                                 + 0.979435 w_k1_MAT_s115
                                 + 1.1315 w_k1_MSFT_s115 + 1.02578 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.940585 w_k1_ADM_s116
                                 + 0.950155 w_k1_AXP_s116
                                 + 1.41381 w_k1_DIS_s116
                                 + 1.03759 w_k1_FDX_s116
                                 + 1.14101 w_k1_IBM_s116
                                 + 1.00541 w_k1_INTC_s116
                                 + 1.12739 w_k1_KO_s116 + 1.09133 w_k1_MAT_s116
                                 + 1.3755 w_k1_MSFT_s116 + 1.04104 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.859281 w_k1_ADM_s116
                                 + 1.04903 w_k1_AXP_s116
                                 + 0.836726 w_k1_DIS_s116
                                 + 0.943188 w_k1_FDX_s116
                                 + 0.958297 w_k1_IBM_s116
                                 + 0.818116 w_k1_INTC_s116
                                 + 0.85327 w_k1_KO_s116 + 1.00906 w_k1_MAT_s116
                                 + 0.845181 w_k1_MSFT_s116
                                 + 1.01065 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.26808 w_k1_ADM_s116
                                 + 1.47419 w_k1_AXP_s116
                                 + 1.58967 w_k1_DIS_s116
                                 + 1.28717 w_k1_FDX_s116
                                 + 1.56165 w_k1_IBM_s116
                                 + 1.06582 w_k1_INTC_s116
                                 + 1.25565 w_k1_KO_s116 + 1.35976 w_k1_MAT_s116
                                 + 1.36357 w_k1_MSFT_s116
                                 + 1.20818 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.807591 w_k1_ADM_s116
                                 + 0.705108 w_k1_AXP_s116
                                 + 0.601263 w_k1_DIS_s116
                                 + 0.728266 w_k1_FDX_s116
                                 + 0.799603 w_k1_IBM_s116
                                 + 0.823519 w_k1_INTC_s116
                                 + 0.684056 w_k1_KO_s116
                                 + 0.754774 w_k1_MAT_s116
                                 + 0.775633 w_k1_MSFT_s116
                                 + 0.825711 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.800553 w_k1_ADM_s116
                                 + 0.625137 w_k1_AXP_s116
                                 + 0.739317 w_k1_DIS_s116
                                 + 0.761883 w_k1_FDX_s116
                                 + 0.802172 w_k1_IBM_s116
                                 + 0.835617 w_k1_INTC_s116
                                 + 0.743487 w_k1_KO_s116
                                 + 0.498133 w_k1_MAT_s116
                                 + 0.969031 w_k1_MSFT_s116
                                 + 0.880353 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.25546 w_k1_ADM_s116
                                 + 1.18855 w_k1_AXP_s116
                                 + 1.24272 w_k1_DIS_s116
                                 + 1.25767 w_k1_FDX_s116
                                 + 0.932678 w_k1_IBM_s116
                                 + 0.939701 w_k1_INTC_s116
                                 + 1.02036 w_k1_KO_s116
                                 + 0.942581 w_k1_MAT_s116
                                 + 1.28869 w_k1_MSFT_s116
                                 + 1.16199 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.923273 w_k1_ADM_s116
                                 + 0.960902 w_k1_AXP_s116
                                 + 1.12239 w_k1_DIS_s116
                                 + 0.911194 w_k1_FDX_s116
                                 + 1.43796 w_k1_IBM_s116
                                 + 0.807778 w_k1_INTC_s116
                                 + 1.24409 w_k1_KO_s116 + 1.38974 w_k1_MAT_s116
                                 + 1.01159 w_k1_MSFT_s116 + 1.0692 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.929606 w_k1_ADM_s116
                                 + 0.852559 w_k1_AXP_s116
                                 + 0.882474 w_k1_DIS_s116
                                 + 1.00564 w_k1_FDX_s116
                                 + 0.681606 w_k1_IBM_s116
                                 + 1.00617 w_k1_INTC_s116
                                 + 0.763805 w_k1_KO_s116
                                 + 0.632964 w_k1_MAT_s116
                                 + 1.07982 w_k1_MSFT_s116
                                 + 1.02368 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.867366 w_k1_ADM_s117
                                 + 0.956322 w_k1_AXP_s117
                                 + 1.0457 w_k1_DIS_s117
                                 + 0.951137 w_k1_FDX_s117
                                 + 1.0432 w_k1_IBM_s117
                                 + 1.07036 w_k1_INTC_s117
                                 + 1.22052 w_k1_KO_s117
                                 + 0.877513 w_k1_MAT_s117
                                 + 1.02087 w_k1_MSFT_s117
                                 + 1.13862 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.51833 w_k1_ADM_s117
                                 + 1.38738 w_k1_AXP_s117
                                 + 1.74124 w_k1_DIS_s117
                                 + 1.32642 w_k1_FDX_s117
                                 + 1.53604 w_k1_IBM_s117
                                 + 1.22781 w_k1_INTC_s117
                                 + 1.71058 w_k1_KO_s117 + 1.53132 w_k1_MAT_s117
                                 + 1.27837 w_k1_MSFT_s117
                                 + 1.39383 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.782701 w_k1_ADM_s117
                                 + 0.645353 w_k1_AXP_s117
                                 + 0.672139 w_k1_DIS_s117
                                 + 0.793199 w_k1_FDX_s117
                                 + 0.796236 w_k1_IBM_s117
                                 + 1.03719 w_k1_INTC_s117
                                 + 0.858577 w_k1_KO_s117
                                 + 0.536223 w_k1_MAT_s117
                                 + 0.747777 w_k1_MSFT_s117
                                 + 1.00235 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.863953 w_k1_ADM_s117
                                 + 0.99832 w_k1_AXP_s117
                                 + 0.674545 w_k1_DIS_s117
                                 + 0.94773 w_k1_FDX_s117
                                 + 0.606829 w_k1_IBM_s117
                                 + 0.941247 w_k1_INTC_s117
                                 + 0.803799 w_k1_KO_s117
                                 + 0.688499 w_k1_MAT_s117
                                 + 0.799259 w_k1_MSFT_s117
                                 + 0.951403 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.387 w_k1_ADM_s117
                                 + 1.26254 w_k1_AXP_s117
                                 + 1.21579 w_k1_DIS_s117
                                 + 1.18767 w_k1_FDX_s117
                                 + 1.05032 w_k1_IBM_s117
                                 + 1.21113 w_k1_INTC_s117
                                 + 1.22874 w_k1_KO_s117 + 1.06532 w_k1_MAT_s117
                                 + 1.10099 w_k1_MSFT_s117
                                 + 1.17926 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.980845 w_k1_ADM_s117
                                 + 0.941461 w_k1_AXP_s117
                                 + 0.654764 w_k1_DIS_s117
                                 + 0.787309 w_k1_FDX_s117
                                 + 1.09025 w_k1_IBM_s117
                                 + 0.905923 w_k1_INTC_s117
                                 + 0.946371 w_k1_KO_s117 + 1.0122 w_k1_MAT_s117
                                 + 0.702646 w_k1_MSFT_s117
                                 + 0.861264 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.996944 w_k1_ADM_s117
                                 + 1.09912 w_k1_AXP_s117
                                 + 0.860663 w_k1_DIS_s117
                                 + 0.949292 w_k1_FDX_s117
                                 + 0.930665 w_k1_IBM_s117
                                 + 1.18547 w_k1_INTC_s117
                                 + 0.969657 w_k1_KO_s117
                                 + 1.13576 w_k1_MAT_s117
                                 + 0.859271 w_k1_MSFT_s117
                                 + 1.00268 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.13713 w_k1_ADM_s117
                                 + 1.28322 w_k1_AXP_s117
                                 + 0.927401 w_k1_DIS_s117
                                 + 0.909641 w_k1_FDX_s117
                                 + 1.5939 w_k1_IBM_s117
                                 + 1.08082 w_k1_INTC_s117
                                 + 1.31259 w_k1_KO_s117 + 1.30673 w_k1_MAT_s117
                                 + 0.833619 w_k1_MSFT_s117
                                 + 1.11236 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0266917386332338 w_k1_ADM_s000
                                 + 0.00950026105073579 w_k1_ADM_s001
                                 + 0.0251916915500877 w_k1_ADM_s002
                                 + 0.0120935861641783 w_k1_ADM_s003
                                 + 0.0075374081479906 w_k1_ADM_s004
                                 + 0.0332477208367212 w_k1_ADM_s005
                                 + 0.00704289866661632 w_k1_ADM_s006
                                 + 0.00943779871050833 w_k1_ADM_s007
                                 + 0.0369531123937626 w_k1_ADM_s010
                                 + 0.053345701763827 w_k1_ADM_s011
                                 + 0.0173319154580879 w_k1_ADM_s012
                                 + 0.0589938919086643 w_k1_ADM_s013
                                 + 0.0199205013892541 w_k1_ADM_s014
                                 + 0.0453865291300322 w_k1_ADM_s015
                                 + 0.0255940957697289 w_k1_ADM_s016
                                 + 0.0282720837472075 w_k1_ADM_s017
                                 + 0.0142410732491738 w_k1_ADM_s100
                                 + 0.0135117839316674 w_k1_ADM_s101
                                 + 0.0119579885785276 w_k1_ADM_s102
                                 + 0.0244398476452003 w_k1_ADM_s103
                                 + 0.0224985205112457 w_k1_ADM_s104
                                 + 0.0584847986693453 w_k1_ADM_s105
                                 + 0.0239180729955473 w_k1_ADM_s106
                                 + 0.0186909577115608 w_k1_ADM_s107
                                 + 0.0829443411575427 w_k1_ADM_s110
                                 + 0.0382318093040496 w_k1_ADM_s111
                                 + 0.0295843174817119 w_k1_ADM_s112
                                 + 0.0275828809173353 w_k1_ADM_s113
                                 + 0.037388636507522 w_k1_ADM_s114
                                 + 0.0690027831949638 w_k1_ADM_s115
                                 + 0.0937254333028969 w_k1_ADM_s116
                                 + 0.0363649692998442 w_k1_ADM_s117
                                 + 0.0228102406966639 w_k1_AXP_s000
                                 + 0.00903895363434415 w_k1_AXP_s001
                                 + 0.030511139855973 w_k1_AXP_s002
                                 + 0.0128662432037939 w_k1_AXP_s003
                                 + 0.00818782498075792 w_k1_AXP_s004
                                 + 0.0330282129868167 w_k1_AXP_s005
                                 + 0.00618593447646705 w_k1_AXP_s006
                                 + 0.00861941591229821 w_k1_AXP_s007
                                 + 0.0379394210770923 w_k1_AXP_s010
                                 + 0.0553356296070135 w_k1_AXP_s011
                                 + 0.0189705643013457 w_k1_AXP_s012
                                 + 0.0474765314380707 w_k1_AXP_s013
                                 + 0.0206009881191307 w_k1_AXP_s014
                                 + 0.0374906970952053 w_k1_AXP_s015
                                 + 0.0224143413695329 w_k1_AXP_s016
                                 + 0.0311610667116941 w_k1_AXP_s017
                                 + 0.0140303762076205 w_k1_AXP_s100
                                 + 0.0149900662382031 w_k1_AXP_s101
                                 + 0.0128137543293311 w_k1_AXP_s102
                                 + 0.0219615836847006 w_k1_AXP_s103
                                 + 0.0182278469545048 w_k1_AXP_s104
                                 + 0.0594154187533221 w_k1_AXP_s105
                                 + 0.0221652735529484 w_k1_AXP_s106
                                 + 0.0187720519746417 w_k1_AXP_s107
                                 + 0.083398637942489 w_k1_AXP_s110
                                 + 0.0402600098865606 w_k1_AXP_s111
                                 + 0.030693330095117 w_k1_AXP_s112
                                 + 0.0280146052189736 w_k1_AXP_s113
                                 + 0.0355575983870287 w_k1_AXP_s114
                                 + 0.0730623387060951 w_k1_AXP_s115
                                 + 0.0966065771816837 w_k1_AXP_s116
                                 + 0.0358165832357012 w_k1_AXP_s117
                                 + 0.0226290136309662 w_k1_DIS_s000
                                 + 0.00877062935372999 w_k1_DIS_s001
                                 + 0.0329409315779397 w_k1_DIS_s002
                                 + 0.0135960562926709 w_k1_DIS_s003
                                 + 0.00812340493596807 w_k1_DIS_s004
                                 + 0.0316797355593506 w_k1_DIS_s005
                                 + 0.00701202058890929 w_k1_DIS_s006
                                 + 0.00881360024759364 w_k1_DIS_s007
                                 + 0.0399400101212714 w_k1_DIS_s010
                                 + 0.0523556625087297 w_k1_DIS_s011
                                 + 0.0199409836047892 w_k1_DIS_s012
                                 + 0.0458930898956627 w_k1_DIS_s013
                                 + 0.020895392490499 w_k1_DIS_s014
                                 + 0.0402498113252182 w_k1_DIS_s015
                                 + 0.024235604797332 w_k1_DIS_s016
                                 + 0.0307998348179411 w_k1_DIS_s017
                                 + 0.0138167905093264 w_k1_DIS_s100
                                 + 0.0146235028668048 w_k1_DIS_s101
                                 + 0.0119691266712199 w_k1_DIS_s102
                                 + 0.0228836038738154 w_k1_DIS_s103
                                 + 0.0195736995893516 w_k1_DIS_s104
                                 + 0.0600510880881604 w_k1_DIS_s105
                                 + 0.0241447189214397 w_k1_DIS_s106
                                 + 0.021008425622866 w_k1_DIS_s107
                                 + 0.0857808480277732 w_k1_DIS_s110
                                 + 0.0405943603517553 w_k1_DIS_s111
                                 + 0.0323054785067209 w_k1_DIS_s112
                                 + 0.0273943748337775 w_k1_DIS_s113
                                 + 0.0363075809824248 w_k1_DIS_s114
                                 + 0.0760763091350611 w_k1_DIS_s115
                                 + 0.100516570002652 w_k1_DIS_s116
                                 + 0.0339091697298902 w_k1_DIS_s117
                                 + 0.0242758511963661 w_k1_FDX_s000
                                 + 0.00886679172103154 w_k1_FDX_s001
                                 + 0.0322321918166488 w_k1_FDX_s002
                                 + 0.0138260217127793 w_k1_FDX_s003
                                 + 0.00828828298921987 w_k1_FDX_s004
                                 + 0.0313776752978615 w_k1_FDX_s005
                                 + 0.00693500599459025 w_k1_FDX_s006
                                 + 0.00853999534913907 w_k1_FDX_s007
                                 + 0.038837207521646 w_k1_FDX_s010
                                 + 0.0538959010986163 w_k1_FDX_s011
                                 + 0.0200581486001026 w_k1_FDX_s012
                                 + 0.0463244732823309 w_k1_FDX_s013
                                 + 0.018900310159733 w_k1_FDX_s014
                                 + 0.0376942777584034 w_k1_FDX_s015
                                 + 0.0245276990202831 w_k1_FDX_s016
                                 + 0.030975687704983 w_k1_FDX_s017
                                 + 0.0134683023367352 w_k1_FDX_s100
                                 + 0.0141389242888744 w_k1_FDX_s101
                                 + 0.0119261880198492 w_k1_FDX_s102
                                 + 0.0226580630392878 w_k1_FDX_s103
                                 + 0.0192623185908019 w_k1_FDX_s104
                                 + 0.0593500759604638 w_k1_FDX_s105
                                 + 0.023007221805568 w_k1_FDX_s106
                                 + 0.0198901575993975 w_k1_FDX_s107
                                 + 0.0854365135896943 w_k1_FDX_s110
                                 + 0.0419077874372017 w_k1_FDX_s111
                                 + 0.0311578617523532 w_k1_FDX_s112
                                 + 0.0278022937986218 w_k1_FDX_s113
                                 + 0.0378305966618986 w_k1_FDX_s114
                                 + 0.0713881855845274 w_k1_FDX_s115
                                 + 0.0961108429927948 w_k1_FDX_s116
                                 + 0.0339387874936698 w_k1_FDX_s117
                                 + 0.0249315687730584 w_k1_IBM_s000
                                 + 0.00896640305825889 w_k1_IBM_s001
                                 + 0.0321967799685871 w_k1_IBM_s002
                                 + 0.0143118741661443 w_k1_IBM_s003
                                 + 0.00843332881360936 w_k1_IBM_s004
                                 + 0.0306401751862034 w_k1_IBM_s005
                                 + 0.00779651602414067 w_k1_IBM_s006
                                 + 0.00871190102088976 w_k1_IBM_s007
                                 + 0.0395861825844506 w_k1_IBM_s010
                                 + 0.0532855252540567 w_k1_IBM_s011
                                 + 0.0199107406493541 w_k1_IBM_s012
                                 + 0.0474760292450176 w_k1_IBM_s013
                                 + 0.0191520387943197 w_k1_IBM_s014
                                 + 0.0402316825269374 w_k1_IBM_s015
                                 + 0.0259604933407652 w_k1_IBM_s016
                                 + 0.0313740438139855 w_k1_IBM_s017
                                 + 0.0143273477213811 w_k1_IBM_s100
                                 + 0.0142186395522312 w_k1_IBM_s101
                                 + 0.0125812762599125 w_k1_IBM_s102
                                 + 0.021090150371909 w_k1_IBM_s103
                                 + 0.0188097922841683 w_k1_IBM_s104
                                 + 0.059560195330122 w_k1_IBM_s105
                                 + 0.0236603032095277 w_k1_IBM_s106
                                 + 0.0204677655467046 w_k1_IBM_s107
                                 + 0.0860565833080252 w_k1_IBM_s110
                                 + 0.0408714783896676 w_k1_IBM_s111
                                 + 0.0335387185937864 w_k1_IBM_s112
                                 + 0.0276688747619698 w_k1_IBM_s113
                                 + 0.037780314093114 w_k1_IBM_s114
                                 + 0.0717232131493904 w_k1_IBM_s115
                                 + 0.102446291770981 w_k1_IBM_s116
                                 + 0.0347133229704532 w_k1_IBM_s117
                                 + 0.0231216724389418 w_k1_INTC_s000
                                 + 0.00843633756286424 w_k1_INTC_s001
                                 + 0.0371260536601316 w_k1_INTC_s002
                                 + 0.0138379179208882 w_k1_INTC_s003
                                 + 0.00882279184162564 w_k1_INTC_s004
                                 + 0.0277068770043392 w_k1_INTC_s005
                                 + 0.00738822256433831 w_k1_INTC_s006
                                 + 0.00869395289670601 w_k1_INTC_s007
                                 + 0.0375305615039832 w_k1_INTC_s010
                                 + 0.0601443844384948 w_k1_INTC_s011
                                 + 0.0202983413548041 w_k1_INTC_s012
                                 + 0.0460097401718852 w_k1_INTC_s013
                                 + 0.0171886834017139 w_k1_INTC_s014
                                 + 0.0356678174572652 w_k1_INTC_s015
                                 + 0.0236546104630323 w_k1_INTC_s016
                                 + 0.0284580676922033 w_k1_INTC_s017
                                 + 0.0169583981429982 w_k1_INTC_s100
                                 + 0.014918662260262 w_k1_INTC_s101
                                 + 0.011937028806681 w_k1_INTC_s102
                                 + 0.0217309699855154 w_k1_INTC_s103
                                 + 0.0185195098927373 w_k1_INTC_s104
                                 + 0.0652644879664594 w_k1_INTC_s105
                                 + 0.0212858225544948 w_k1_INTC_s106
                                 + 0.0198174745704946 w_k1_INTC_s107
                                 + 0.0819262315188535 w_k1_INTC_s110
                                 + 0.0428379560334499 w_k1_INTC_s111
                                 + 0.0353769363631175 w_k1_INTC_s112
                                 + 0.0243905514249788 w_k1_INTC_s113
                                 + 0.0352203784838424 w_k1_INTC_s114
                                 + 0.0758414309314129 w_k1_INTC_s115
                                 + 0.0859387132012503 w_k1_INTC_s116
                                 + 0.0372949061131132 w_k1_INTC_s117
                                 + 0.023771954671541 w_k1_KO_s000
                                 + 0.00807605793377467 w_k1_KO_s001
                                 + 0.033077002427624 w_k1_KO_s002
                                 + 0.0147207160311685 w_k1_KO_s003
                                 + 0.00800500770646088 w_k1_KO_s004
                                 + 0.0309830823246133 w_k1_KO_s005
                                 + 0.00767776102427528 w_k1_KO_s006
                                 + 0.00890293677747809 w_k1_KO_s007
                                 + 0.0411318490088833 w_k1_KO_s010
                                 + 0.0515104369834044 w_k1_KO_s011
                                 + 0.0187858792610255 w_k1_KO_s012
                                 + 0.044757267966956 w_k1_KO_s013
                                 + 0.019932420534083 w_k1_KO_s014
                                 + 0.0374118162437478 w_k1_KO_s015
                                 + 0.025790854852305 w_k1_KO_s016
                                 + 0.0293558765817803 w_k1_KO_s017
                                 + 0.0144194505143824 w_k1_KO_s100
                                 + 0.0139729004171365 w_k1_KO_s101
                                 + 0.0133663511939258 w_k1_KO_s102
                                 + 0.0213054328668982 w_k1_KO_s103
                                 + 0.0173789198081853 w_k1_KO_s104
                                 + 0.0583782517703615 w_k1_KO_s105
                                 + 0.0241641969723868 w_k1_KO_s106
                                 + 0.0205598787902722 w_k1_KO_s107
                                 + 0.0875868774423694 w_k1_KO_s110
                                 + 0.0436494041112448 w_k1_KO_s111
                                 + 0.0335478116364586 w_k1_KO_s112
                                 + 0.0237964822345168 w_k1_KO_s113
                                 + 0.0386457159962396 w_k1_KO_s114
                                 + 0.0842618679885031 w_k1_KO_s115
                                 + 0.0945473593249744 w_k1_KO_s116
                                 + 0.0381789497358483 w_k1_KO_s117
                                 + 0.0210600110904505 w_k1_MAT_s000
                                 + 0.00815490352222686 w_k1_MAT_s001
                                 + 0.0294848160212238 w_k1_MAT_s002
                                 + 0.0140268022613588 w_k1_MAT_s003
                                 + 0.00750981446708212 w_k1_MAT_s004
                                 + 0.0281368094870695 w_k1_MAT_s005
                                 + 0.00675157924520505 w_k1_MAT_s006
                                 + 0.00727390589050171 w_k1_MAT_s007
                                 + 0.0429233040800733 w_k1_MAT_s010
                                 + 0.0573521650396323 w_k1_MAT_s011
                                 + 0.0187552360498111 w_k1_MAT_s012
                                 + 0.053703607925452 w_k1_MAT_s013
                                 + 0.021874946816416 w_k1_MAT_s014
                                 + 0.0449952335560557 w_k1_MAT_s015
                                 + 0.0273527019732394 w_k1_MAT_s016
                                 + 0.0392681178706126 w_k1_MAT_s017
                                 + 0.0149537060586166 w_k1_MAT_s100
                                 + 0.0135798563942699 w_k1_MAT_s101
                                 + 0.0109374611793735 w_k1_MAT_s102
                                 + 0.0241522576825773 w_k1_MAT_s103
                                 + 0.0159302849044834 w_k1_MAT_s104
                                 + 0.0786590084975302 w_k1_MAT_s105
                                 + 0.0216321934691037 w_k1_MAT_s106
                                 + 0.0226380404065996 w_k1_MAT_s107
                                 + 0.0876834988121101 w_k1_MAT_s110
                                 + 0.0415821523707757 w_k1_MAT_s111
                                 + 0.0327384094358367 w_k1_MAT_s112
                                 + 0.0260811111391312 w_k1_MAT_s113
                                 + 0.0427049906448344 w_k1_MAT_s114
                                 + 0.0919239994443062 w_k1_MAT_s115
                                 + 0.0979650346023209 w_k1_MAT_s116
                                 + 0.0326440140797149 w_k1_MAT_s117
                                 + 0.0235078613862681 w_k1_MSFT_s000
                                 + 0.00885101590718807 w_k1_MSFT_s001
                                 + 0.032592168585936 w_k1_MSFT_s002
                                 + 0.013141757436725 w_k1_MSFT_s003
                                 + 0.00850209755965348 w_k1_MSFT_s004
                                 + 0.029407306273352 w_k1_MSFT_s005
                                 + 0.00757643332705464 w_k1_MSFT_s006
                                 + 0.00803177030195826 w_k1_MSFT_s007
                                 + 0.0399360607305165 w_k1_MSFT_s010
                                 + 0.0561732362327855 w_k1_MSFT_s011
                                 + 0.0208613903469382 w_k1_MSFT_s012
                                 + 0.0429166227911644 w_k1_MSFT_s013
                                 + 0.0180104984223405 w_k1_MSFT_s014
                                 + 0.0378223884547823 w_k1_MSFT_s015
                                 + 0.0234353263763291 w_k1_MSFT_s016
                                 + 0.0298156951748164 w_k1_MSFT_s017
                                 + 0.0145074628772875 w_k1_MSFT_s100
                                 + 0.0145828909161435 w_k1_MSFT_s101
                                 + 0.0114599686163991 w_k1_MSFT_s102
                                 + 0.022695309473172 w_k1_MSFT_s103
                                 + 0.0188114047346621 w_k1_MSFT_s104
                                 + 0.0630132013373541 w_k1_MSFT_s105
                                 + 0.0235278653826955 w_k1_MSFT_s106
                                 + 0.019637390828167 w_k1_MSFT_s107
                                 + 0.0829436213110175 w_k1_MSFT_s110
                                 + 0.043149467224221 w_k1_MSFT_s111
                                 + 0.0324151720106675 w_k1_MSFT_s112
                                 + 0.0249194185266966 w_k1_MSFT_s113
                                 + 0.0351626875072339 w_k1_MSFT_s114
                                 + 0.0787677692590142 w_k1_MSFT_s115
                                 + 0.101846589343533 w_k1_MSFT_s116
                                 + 0.0322469202678982 w_k1_MSFT_s117
                                 + 0.0258945963941114 w_k1_PG_s000
                                 + 0.0089964647457883 w_k1_PG_s001
                                 + 0.0324777843435597 w_k1_PG_s002
                                 + 0.0135175945897798 w_k1_PG_s003
                                 + 0.00857449456851018 w_k1_PG_s004
                                 + 0.0307054862330066 w_k1_PG_s005
                                 + 0.00717525468599286 w_k1_PG_s006
                                 + 0.0078030472262056 w_k1_PG_s007
                                 + 0.0362968720666339 w_k1_PG_s010
                                 + 0.0554016236184801 w_k1_PG_s011
                                 + 0.0214597407093882 w_k1_PG_s012
                                 + 0.0449696165294685 w_k1_PG_s013
                                 + 0.0192023499455299 w_k1_PG_s014
                                 + 0.0397382931941926 w_k1_PG_s015
                                 + 0.0231089587979917 w_k1_PG_s016
                                 + 0.0320649690396169 w_k1_PG_s017
                                 + 0.0140050587411306 w_k1_PG_s100
                                 + 0.0142587230640434 w_k1_PG_s101
                                 + 0.011694602662492 w_k1_PG_s102
                                 + 0.0224236119733416 w_k1_PG_s103
                                 + 0.0196576752868401 w_k1_PG_s104
                                 + 0.0569152984658678 w_k1_PG_s105
                                 + 0.0222703642010096 w_k1_PG_s106
                                 + 0.0205154192699031 w_k1_PG_s107
                                 + 0.0822900490435989 w_k1_PG_s110
                                 + 0.0414063905208289 w_k1_PG_s111
                                 + 0.0310579944993645 w_k1_PG_s112
                                 + 0.0282245525149108 w_k1_PG_s113
                                 + 0.0418269580817777 w_k1_PG_s114
                                 + 0.0708839555551582 w_k1_PG_s115
                                 + 0.100277884656353 w_k1_PG_s116
                                 + 0.0370671320784524 w_k1_PG_s117 >= 1.17
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
