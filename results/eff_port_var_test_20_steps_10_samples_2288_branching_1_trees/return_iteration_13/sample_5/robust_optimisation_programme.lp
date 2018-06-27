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
 w_k1_ADM_s0_return_constraint:  1.09563 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.952382 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.05474 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.33815 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.88142 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.00719 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.971226 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.06133 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.759423 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.780539 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.743793 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.44151 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.737575 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.02006 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.888569 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.09447 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.831892 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.781081 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.820501 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.57958 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.19614 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.65091 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.932081 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.25893 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.02892 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.860279 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.07128 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.49236 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.07761 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.803068 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.69845 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.31502 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.14816 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.859895 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.924624 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.28896 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.925019 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.10024 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.21265 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.864284 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.89201 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.34741 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.903503 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.11973 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.11829 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.08741 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.864644 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.727562 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.837118 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.72261 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.64573 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.34517 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.24495 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.08323 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.32784 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.759719 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.920653 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.670201 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.24784 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.765468 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.782094 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.32605 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.12757 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.72666 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.07251 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.84561 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.4472 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.696562 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.604357 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.38046 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.5308 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.969888 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.827271 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.23354 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.918867 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.17774 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.25074 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.829557 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.934241 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.42193 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.90972 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.15266 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.37397 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.932925 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.795278 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.565166 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.800841 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.97887 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.573774 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.12132 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.22853 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.86819 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.25719 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.767431 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.06133 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.552005 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.64262 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.700913 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.787362 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.39894 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.905239 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.832563 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.00494 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 2.05196 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.29327 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.652964 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.540478 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.42382 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.18376 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.00293 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.750738 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.58709 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.891984 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.860244 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.1394 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.871774 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.972941 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.18043 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.96469 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.04178 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.18823 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.07512 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.831048 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.682571 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.791178 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.61178 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.743116 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.05557 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.1064 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.0416 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.13241 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.829226 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.09672 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.671179 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.41468 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.74904 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.901188 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.35152 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.954726 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.811 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.082 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.70072 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.1939 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.749019 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.650426 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.26266 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.26901 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.916285 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.869638 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.39437 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.00001 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.12491 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15057 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.899754 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.890433 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.50703 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.767149 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.25737 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.03832 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.821473 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.883643 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.05182 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.40107 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.14589 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.603415 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.65856 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.13032 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.836783 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.65818 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.766059 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.851038 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.855987 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.928797 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.19812 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.697097 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.0571 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.24927 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.55903 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.854363 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.19185 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.48868 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.964583 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.760319 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.45149 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.09843 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.55685 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.985367 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.912193 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.765246 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.696099 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.10021 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.887568 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.943922 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.13467 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.915875 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.1465 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.14154 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.990881 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.905396 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.762801 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.04685 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.54031 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.796762 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.24618 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.05262 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.963159 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.07674 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.919252 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.12276 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.818567 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.20211 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.864731 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.971241 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.25303 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.918548 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.953683 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.03402 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.38296 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.38403 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.896655 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.620707 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.2031 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.992435 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.930699 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.791887 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.14586 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.924356 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.93418 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.13877 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.889382 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.961069 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.43034 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.826471 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.23087 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.15364 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.852824 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.891267 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.82478 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.03377 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.37096 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.61706 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.22518 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.16581 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.893077 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.15988 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.801864 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.02388 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.784249 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.09651 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.891724 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.786605 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.15972 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.05445 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.26597 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.892183 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.417 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.33014 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.878141 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.688537 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.35057 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.9657 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.1285 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.806868 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.10075 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.791914 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.731595 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.26224 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.844503 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.774016 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.57478 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.745078 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.34198 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.20651 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.15664 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.736885 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.00476 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.4017 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.37971 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.556814 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.91298 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.23892 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.977495 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.70517 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.577217 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.824816 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.87084 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.924 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.03266 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.656764 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.14306 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.56981 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.2773 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.2063 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.28346 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.93894 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.751061 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.637398 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.71423 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.17028 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.29001 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.873511 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.788872 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.858032 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.90822 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.13544 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.85663 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.915562 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.18941 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.01724 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.14196 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.22015 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.07962 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.904111 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.77611 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.884062 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.55818 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.763318 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.08636 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.08955 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.892139 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.08511 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.893246 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.09389 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.68976 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.22428 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.7722 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.852698 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.244 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.909471 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.883981 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.989348 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.53498 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.22017 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.767096 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.670813 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.21608 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.00645 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.890206 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.811991 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.40588 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.89134 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.912814 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12725 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.899419 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.967488 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.23245 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.974693 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.05985 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.09781 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.917443 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.899749 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.786042 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.917586 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.35209 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.722191 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.0437 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.05549 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.914304 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.03686 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.818285 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.986386 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.795587 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.10783 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.952481 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.835527 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.07163 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.02758 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.936179 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.939891 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.19525 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.09044 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.853969 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.808032 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.22474 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.07739 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.0284 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.942991 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.19044 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.990675 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.977144 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 1.43600181679628 z_k1_s0000
                                 - 1.37999993731912 z_k1_s0001
                                 - 1.2320052764335 z_k1_s0002
                                 - 0.631001402459327 z_k1_s0003
                                 - 0.372000375191067 z_k1_s0004
                                 - 0.948000897951326 z_k1_s0005
                                 - 1.28799813789464 z_k1_s0006
                                 - 1.73099915680545 z_k1_s0007
                                 - 0.849002489414592 z_k1_s0010
                                 - 1.3449996386269 z_k1_s0011
                                 - 0.910002814068899 z_k1_s0012
                                 - 1.06899828906008 z_k1_s0013
                                 - 0.974002915968083 z_k1_s0014
                                 - 0.937000809272787 z_k1_s0015
                                 - 0.538000583552032 z_k1_s0016
                                 - 0.659000821711505 z_k1_s0017
                                 - 0.631999860128997 z_k1_s0020
                                 - 1.22499831071287 z_k1_s0021
                                 - 1.34699803428999 z_k1_s0022
                                 - 0.491999754261816 z_k1_s0023
                                 - 0.62000007026895 z_k1_s0024
                                 - 0.996002088383646 z_k1_s0025
                                 - 1.05399921270723 z_k1_s0026
                                 - 0.608999797897246 z_k1_s0027
                                 - 0.129999966627623 z_k1_s0030
                                 - 0.150000001985707 z_k1_s0031
                                 - 0.581000658672434 z_k1_s0032
                                 - 0.333000536072103 z_k1_s0033
                                 - 0.251999971751997 z_k1_s0034
                                 - 0.476999843130598 z_k1_s0035
                                 - 0.22799982404233 z_k1_s0036
                                 - 0.481000481882017 z_k1_s0037
                                 - 0.182000237569134 z_k1_s0040
                                 - 0.436001041117695 z_k1_s0041
                                 - 0.234000268031709 z_k1_s0042
                                 - 0.179000170336618 z_k1_s0043
                                 - 0.505001277964368 z_k1_s0044
                                 - 0.460001578977822 z_k1_s0045
                                 - 0.286000298494285 z_k1_s0046
                                 - 0.337000131945862 z_k1_s0047
                                 - 0.596001166239636 z_k1_s0050
                                 - 0.429000602115488 z_k1_s0051
                                 - 0.272000091182602 z_k1_s0052
                                 - 0.294000031383414 z_k1_s0053
                                 - 0.359000524476593 z_k1_s0054
                                 - 0.208000104398454 z_k1_s0055
                                 - 0.287000023619525 z_k1_s0056
                                 - 0.502000725181756 z_k1_s0057
                                 - 0.322000097449719 z_k1_s0060
                                 - 0.341999992011788 z_k1_s0061
                                 - 0.54400000408893 z_k1_s0062
                                 - 0.219999917183004 z_k1_s0063
                                 - 0.777999847465383 z_k1_s0064
                                 - 0.298000223975237 z_k1_s0065
                                 - 0.776000217009257 z_k1_s0066
                                 - 0.310000160712478 z_k1_s0067
                                 - 0.395999009804553 z_k1_s0070
                                 - 0.972999637171989 z_k1_s0071
                                 - 0.427998650539046 z_k1_s0072
                                 - 0.505998515339773 z_k1_s0073
                                 - 0.40499910270433 z_k1_s0074
                                 - 0.477998477073091 z_k1_s0075
                                 - 1.11999588868326 z_k1_s0076
                                 - 1.33599360469068 z_k1_s0077
                                 - 0.145999825355207 z_k1_s0100
                                 - 0.174999541172769 z_k1_s0101
                                 - 0.0529999250864992 z_k1_s0102
                                 - 0.117000109537645 z_k1_s0103
                                 - 0.213999577685318 z_k1_s0104
                                 - 0.079999856964579 z_k1_s0105
                                 - 0.0539998637561595 z_k1_s0106
                                 - 0.0949997776080285 z_k1_s0107
                                 - 0.230999192130292 z_k1_s0110
                                 - 0.145999618898043 z_k1_s0111
                                 - 0.372998436518391 z_k1_s0112
                                 - 0.368998968506183 z_k1_s0113
                                 - 0.263999335719687 z_k1_s0114
                                 - 0.0949996936591463 z_k1_s0115
                                 - 0.188000378510378 z_k1_s0116
                                 - 0.146999667200642 z_k1_s0117
                                 - 0.111999896989828 z_k1_s0120
                                 - 0.287999735116702 z_k1_s0121
                                 - 0.172999674136186 z_k1_s0122
                                 - 0.225000293309588 z_k1_s0123
                                 - 0.0829999570120827 z_k1_s0124
                                 - 0.147000031549038 z_k1_s0125
                                 - 0.158999353512681 z_k1_s0126
                                 - 0.147000031549038 z_k1_s0127
                                 - 0.257000615387667 z_k1_s0130
                                 - 0.208000053344727 z_k1_s0131
                                 - 0.249000242605048 z_k1_s0132
                                 - 0.0939999737942343 z_k1_s0133
                                 - 0.136000003102311 z_k1_s0134
                                 - 0.110000030859232 z_k1_s0135
                                 - 0.0790000321770881 z_k1_s0136
                                 - 0.244000353866032 z_k1_s0137
                                 - 0.229999217684282 z_k1_s0140
                                 - 0.0549999235851276 z_k1_s0141
                                 - 0.0539998439910312 z_k1_s0142
                                 - 0.103999369605677 z_k1_s0143
                                 - 0.0960001872496973 z_k1_s0144
                                 - 0.162999702466989 z_k1_s0145
                                 - 0.123000154695113 z_k1_s0146
                                 - 0.0839997775193372 z_k1_s0147
                                 - 0.114999763262807 z_k1_s0150
                                 - 0.204999619033626 z_k1_s0151
                                 - 0.191999818177523 z_k1_s0152
                                 - 0.169999937344507 z_k1_s0153
                                 - 0.0649998538789996 z_k1_s0154
                                 - 0.0679998805379709 z_k1_s0155
                                 - 0.0509998868035202 z_k1_s0156
                                 - 0.0779998435347659 z_k1_s0157
                                 - 0.229999807147226 z_k1_s0160
                                 - 0.209999812595309 z_k1_s0161
                                 - 0.56199971670905 z_k1_s0162
                                 - 0.178999755939926 z_k1_s0163
                                 - 0.477999999990641 z_k1_s0164
                                 - 0.300998550908229 z_k1_s0165
                                 - 0.42900059923764 z_k1_s0166
                                 - 0.214999746133377 z_k1_s0167
                                 - 0.077999870782309 z_k1_s0170
                                 - 0.0579999177611854 z_k1_s0171
                                 - 0.23199988704445 z_k1_s0172
                                 - 0.0969998531523399 z_k1_s0173
                                 - 0.0989999024543794 z_k1_s0174
                                 - 0.283999656899517 z_k1_s0175
                                 - 0.0899998425949831 z_k1_s0176
                                 - 0.141999288450487 z_k1_s0177
                                 - 0.409000624823065 z_k1_s1000
                                 - 0.377000736194672 z_k1_s1001
                                 - 0.397000873137812 z_k1_s1002
                                 - 1.73100246127086 z_k1_s1003
                                 - 0.628001325688926 z_k1_s1004
                                 - 0.56100188591135 z_k1_s1005
                                 - 0.734002987849354 z_k1_s1006
                                 - 0.671002363698095 z_k1_s1007
                                 - 0.375000758105736 z_k1_s1010
                                 - 0.557998428545164 z_k1_s1011
                                 - 0.391001015411262 z_k1_s1012
                                 - 0.623999489930457 z_k1_s1013
                                 - 0.332999379679728 z_k1_s1014
                                 - 0.805998901785797 z_k1_s1015
                                 - 0.198999685242461 z_k1_s1016
                                 - 0.22899981619082 z_k1_s1017
                                 - 0.741999429845652 z_k1_s1020
                                 - 0.218999563166348 z_k1_s1021
                                 - 0.251999021156831 z_k1_s1022
                                 - 0.219999568996314 z_k1_s1023
                                 - 0.291000472522862 z_k1_s1024
                                 - 0.196999679706557 z_k1_s1025
                                 - 0.242999458286072 z_k1_s1026
                                 - 0.283999207715754 z_k1_s1027
                                 - 0.279000852770347 z_k1_s1030
                                 - 0.213000651039728 z_k1_s1031
                                 - 0.196000141411508 z_k1_s1032
                                 - 0.163000040546198 z_k1_s1033
                                 - 0.133000223450843 z_k1_s1034
                                 - 0.187000113902787 z_k1_s1035
                                 - 0.0600000460912818 z_k1_s1036
                                 - 0.141999976359182 z_k1_s1037
                                 - 0.266998022315421 z_k1_s1040
                                 - 0.308998490576429 z_k1_s1041
                                 - 0.279999696415914 z_k1_s1042
                                 - 0.213999437675906 z_k1_s1043
                                 - 0.267999929626069 z_k1_s1044
                                 - 0.587998848674798 z_k1_s1045
                                 - 0.389998843152707 z_k1_s1046
                                 - 0.252999321991177 z_k1_s1047
                                 - 0.140000198618293 z_k1_s1050
                                 - 0.0620000545023575 z_k1_s1051
                                 - 0.180000422652237 z_k1_s1052
                                 - 0.098000139032805 z_k1_s1053
                                 - 0.133999989363047 z_k1_s1054
                                 - 0.207000193299817 z_k1_s1055
                                 - 0.098000139032805 z_k1_s1056
                                 - 0.252000591713132 z_k1_s1057
                                 - 0.206999853529616 z_k1_s1060
                                 - 0.204000379710664 z_k1_s1061
                                 - 0.194000325646394 z_k1_s1062
                                 - 0.144000055325046 z_k1_s1063
                                 - 0.22899961087079 z_k1_s1064
                                 - 0.235000366509791 z_k1_s1065
                                 - 0.225000312445521 z_k1_s1066
                                 - 0.370000192376884 z_k1_s1067
                                 - 0.338000895969265 z_k1_s1070
                                 - 0.451000921301355 z_k1_s1071
                                 - 0.41000120709326 z_k1_s1072
                                 - 0.984001271018899 z_k1_s1073
                                 - 0.831002337510644 z_k1_s1074
                                 - 0.444000970095095 z_k1_s1075
                                 - 0.237000786990765 z_k1_s1076
                                 - 0.3700010794134 z_k1_s1077
                                 - 0.358998350112084 z_k1_s1100
                                 - 0.377000473135857 z_k1_s1101
                                 - 0.658999675195742 z_k1_s1102
                                 - 0.338001194914436 z_k1_s1103
                                 - 0.277999971998289 z_k1_s1104
                                 - 0.797001051053757 z_k1_s1105
                                 - 0.205999781698133 z_k1_s1106
                                 - 0.178999790160526 z_k1_s1107
                                 - 0.226999239245626 z_k1_s1110
                                 - 0.308999155529307 z_k1_s1111
                                 - 0.987998136776981 z_k1_s1112
                                 - 0.955998332216949 z_k1_s1113
                                 - 0.419000569197415 z_k1_s1114
                                 - 0.483000178317479 z_k1_s1115
                                 - 0.392998642499391 z_k1_s1116
                                 - 0.395998832726194 z_k1_s1117
                                 - 0.18599945133809 z_k1_s1120
                                 - 0.153000011881201 z_k1_s1121
                                 - 0.112999924006401 z_k1_s1122
                                 - 0.172000007301744 z_k1_s1123
                                 - 0.163000033849901 z_k1_s1124
                                 - 0.155000572114784 z_k1_s1125
                                 - 0.347000005754103 z_k1_s1126
                                 - 0.25499924780222 z_k1_s1127
                                 - 0.557002274101001 z_k1_s1130
                                 - 0.347001917353288 z_k1_s1131
                                 - 0.276000858569449 z_k1_s1132
                                 - 0.564000871522201 z_k1_s1133
                                 - 0.247000780437546 z_k1_s1134
                                 - 0.400002122856586 z_k1_s1135
                                 - 0.348002149986028 z_k1_s1136
                                 - 0.292000640382914 z_k1_s1137
                                 - 0.416999338614115 z_k1_s1140
                                 - 0.419999406629899 z_k1_s1141
                                 - 0.528999038409038 z_k1_s1142
                                 - 0.756996754023469 z_k1_s1143
                                 - 0.624999085318365 z_k1_s1144
                                 - 0.553998362943047 z_k1_s1145
                                 - 0.401998998535196 z_k1_s1146
                                 - 1.61999413660817 z_k1_s1147
                                 - 0.276999799274482 z_k1_s1150
                                 - 0.573999817741121 z_k1_s1151
                                 - 0.633998107119867 z_k1_s1152
                                 - 0.417000149520718 z_k1_s1153
                                 - 0.635999234198021 z_k1_s1154
                                 - 0.597997554694136 z_k1_s1155
                                 - 0.31799961709901 z_k1_s1156
                                 - 0.492999561532279 z_k1_s1157
                                 - 0.832003700965952 z_k1_s1160
                                 - 0.247000761323337 z_k1_s1161
                                 - 0.720003462298173 z_k1_s1162
                                 - 0.362000831658521 z_k1_s1163
                                 - 0.450001742185196 z_k1_s1164
                                 - 0.276000708645928 z_k1_s1165
                                 - 0.195000530012965 z_k1_s1166
                                 - 0.292000935541857 z_k1_s1167
                                 - 0.0840000454448236 z_k1_s1170
                                 - 0.0740000550347262 z_k1_s1171
                                 - 0.100999966141987 z_k1_s1172
                                 - 0.0980000320189601 z_k1_s1173
                                 - 0.210000428612072 z_k1_s1174
                                 - 0.193000507914909 z_k1_s1175
                                 - 0.170000466971682 z_k1_s1176
                                 - 0.330000313533241 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.939752 w_k1_ADM_s000
                                 + 0.890638 w_k1_AXP_s000
                                 + 0.864443 w_k1_DIS_s000
                                 + 0.899758 w_k1_FDX_s000
                                 + 0.70698 w_k1_IBM_s000
                                 + 0.909204 w_k1_INTC_s000
                                 + 0.848643 w_k1_KO_s000
                                 + 0.671353 w_k1_MAT_s000
                                 + 1.00867 w_k1_MSFT_s000
                                 + 0.886557 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.10763 w_k1_ADM_s000
                                 + 0.953933 w_k1_AXP_s000
                                 + 0.875461 w_k1_DIS_s000
                                 + 0.946514 w_k1_FDX_s000
                                 + 1.13447 w_k1_IBM_s000
                                 + 1.01116 w_k1_INTC_s000
                                 + 1.02919 w_k1_KO_s000 + 1.00278 w_k1_MAT_s000
                                 + 1.01647 w_k1_MSFT_s000
                                 + 0.915978 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.47953 w_k1_ADM_s000
                                 + 1.25369 w_k1_AXP_s000
                                 + 1.43126 w_k1_DIS_s000
                                 + 1.39733 w_k1_FDX_s000 + 1.0337 w_k1_IBM_s000
                                 + 1.20342 w_k1_INTC_s000
                                 + 1.20365 w_k1_KO_s000 + 1.16651 w_k1_MAT_s000
                                 + 1.29109 w_k1_MSFT_s000
                                 + 1.18159 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.783738 w_k1_ADM_s000
                                 + 0.609677 w_k1_AXP_s000
                                 + 0.553683 w_k1_DIS_s000
                                 + 0.736202 w_k1_FDX_s000
                                 + 0.688893 w_k1_IBM_s000
                                 + 0.755969 w_k1_INTC_s000
                                 + 0.711243 w_k1_KO_s000
                                 + 0.547968 w_k1_MAT_s000
                                 + 0.750912 w_k1_MSFT_s000
                                 + 0.689515 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.32114 w_k1_ADM_s000
                                 + 0.861455 w_k1_AXP_s000
                                 + 1.27231 w_k1_DIS_s000
                                 + 0.988999 w_k1_FDX_s000
                                 + 1.13417 w_k1_IBM_s000
                                 + 0.93608 w_k1_INTC_s000 + 1.2225 w_k1_KO_s000
                                 + 1.14873 w_k1_MAT_s000
                                 + 1.10479 w_k1_MSFT_s000
                                 + 1.09103 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.85116 w_k1_ADM_s000
                                 + 0.955547 w_k1_AXP_s000
                                 + 0.647208 w_k1_DIS_s000
                                 + 0.801502 w_k1_FDX_s000
                                 + 1.05544 w_k1_IBM_s000
                                 + 0.982648 w_k1_INTC_s000
                                 + 0.840677 w_k1_KO_s000
                                 + 0.93264 w_k1_MAT_s000
                                 + 0.855129 w_k1_MSFT_s000
                                 + 0.810748 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.13052 w_k1_ADM_s000
                                 + 1.18217 w_k1_AXP_s000
                                 + 0.953546 w_k1_DIS_s000
                                 + 1.09066 w_k1_FDX_s000
                                 + 0.746346 w_k1_IBM_s000
                                 + 1.06662 w_k1_INTC_s000
                                 + 0.875256 w_k1_KO_s000
                                 + 0.684224 w_k1_MAT_s000
                                 + 0.948005 w_k1_MSFT_s000
                                 + 1.00203 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.11101 w_k1_ADM_s000
                                 + 1.38585 w_k1_AXP_s000
                                 + 1.24386 w_k1_DIS_s000
                                 + 1.15022 w_k1_FDX_s000
                                 + 1.55817 w_k1_IBM_s000
                                 + 1.09314 w_k1_INTC_s000
                                 + 1.29491 w_k1_KO_s000 + 1.57218 w_k1_MAT_s000
                                 + 1.1568 w_k1_MSFT_s000 + 1.17844 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.18618 w_k1_ADM_s001
                                 + 1.63693 w_k1_AXP_s001
                                 + 1.17325 w_k1_DIS_s001
                                 + 1.25232 w_k1_FDX_s001 + 1.0928 w_k1_IBM_s001
                                 + 1.18901 w_k1_INTC_s001
                                 + 1.02208 w_k1_KO_s001
                                 + 0.982562 w_k1_MAT_s001
                                 + 1.03824 w_k1_MSFT_s001
                                 + 1.17565 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.06781 w_k1_ADM_s001
                                 + 0.940292 w_k1_AXP_s001
                                 + 1.08964 w_k1_DIS_s001 + 1.0529 w_k1_FDX_s001
                                 + 0.856187 w_k1_IBM_s001
                                 + 1.01057 w_k1_INTC_s001
                                 + 1.07716 w_k1_KO_s001
                                 + 0.881337 w_k1_MAT_s001
                                 + 1.04567 w_k1_MSFT_s001
                                 + 1.01656 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.943944 w_k1_ADM_s001
                                 + 0.727177 w_k1_AXP_s001
                                 + 0.844937 w_k1_DIS_s001
                                 + 0.965154 w_k1_FDX_s001
                                 + 0.920214 w_k1_IBM_s001
                                 + 0.946113 w_k1_INTC_s001
                                 + 0.94512 w_k1_KO_s001
                                 + 0.601094 w_k1_MAT_s001
                                 + 0.87516 w_k1_MSFT_s001
                                 + 0.937963 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.08933 w_k1_ADM_s001
                                 + 1.1874 w_k1_AXP_s001 + 1.16132 w_k1_DIS_s001
                                 + 1.11371 w_k1_FDX_s001 + 1.619 w_k1_IBM_s001
                                 + 1.13447 w_k1_INTC_s001
                                 + 1.28936 w_k1_KO_s001 + 1.30972 w_k1_MAT_s001
                                 + 1.09375 w_k1_MSFT_s001
                                 + 1.22671 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.781105 w_k1_ADM_s001
                                 + 0.746648 w_k1_AXP_s001
                                 + 0.728159 w_k1_DIS_s001
                                 + 0.789489 w_k1_FDX_s001
                                 + 0.8628 w_k1_IBM_s001
                                 + 1.02914 w_k1_INTC_s001
                                 + 0.917563 w_k1_KO_s001
                                 + 0.582772 w_k1_MAT_s001
                                 + 0.799892 w_k1_MSFT_s001
                                 + 0.876333 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.04231 w_k1_ADM_s001
                                 + 1.21891 w_k1_AXP_s001
                                 + 0.836591 w_k1_DIS_s001
                                 + 0.932065 w_k1_FDX_s001
                                 + 1.13545 w_k1_IBM_s001
                                 + 1.06237 w_k1_INTC_s001
                                 + 1.14191 w_k1_KO_s001 + 1.09323 w_k1_MAT_s001
                                 + 0.911077 w_k1_MSFT_s001 + 1.148 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.817305 w_k1_ADM_s001
                                 + 0.663938 w_k1_AXP_s001
                                 + 0.719218 w_k1_DIS_s001
                                 + 0.824143 w_k1_FDX_s001
                                 + 0.550837 w_k1_IBM_s001
                                 + 0.833474 w_k1_INTC_s001
                                 + 0.756733 w_k1_KO_s001
                                 + 0.456941 w_k1_MAT_s001
                                 + 0.888989 w_k1_MSFT_s001
                                 + 0.913334 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.13897 w_k1_ADM_s001
                                 + 1.28661 w_k1_AXP_s001
                                 + 1.33812 w_k1_DIS_s001
                                 + 1.33284 w_k1_FDX_s001
                                 + 0.727326 w_k1_IBM_s001
                                 + 1.0854 w_k1_INTC_s001 + 1.0509 w_k1_KO_s001
                                 + 0.831897 w_k1_MAT_s001
                                 + 1.22693 w_k1_MSFT_s001
                                 + 1.19456 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.789077 w_k1_ADM_s002
                                 + 0.872373 w_k1_AXP_s002
                                 + 0.94423 w_k1_DIS_s002
                                 + 0.868825 w_k1_FDX_s002
                                 + 1.26984 w_k1_IBM_s002
                                 + 1.01963 w_k1_INTC_s002
                                 + 1.00793 w_k1_KO_s002 + 1.21457 w_k1_MAT_s002
                                 + 0.969906 w_k1_MSFT_s002
                                 + 0.919197 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.948113 w_k1_ADM_s002
                                 + 0.657083 w_k1_AXP_s002
                                 + 0.627579 w_k1_DIS_s002
                                 + 0.797887 w_k1_FDX_s002
                                 + 0.862265 w_k1_IBM_s002
                                 + 0.863505 w_k1_INTC_s002
                                 + 0.818486 w_k1_KO_s002
                                 + 1.07643 w_k1_MAT_s002
                                 + 0.829157 w_k1_MSFT_s002
                                 + 0.844971 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.08473 w_k1_ADM_s002
                                 + 1.05724 w_k1_AXP_s002
                                 + 0.945885 w_k1_DIS_s002
                                 + 0.974963 w_k1_FDX_s002
                                 + 0.888127 w_k1_IBM_s002
                                 + 0.834541 w_k1_INTC_s002
                                 + 0.808653 w_k1_KO_s002
                                 + 1.27944 w_k1_MAT_s002
                                 + 1.02677 w_k1_MSFT_s002
                                 + 0.918285 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.10981 w_k1_ADM_s002
                                 + 1.36091 w_k1_AXP_s002 + 1.356 w_k1_DIS_s002
                                 + 1.19086 w_k1_FDX_s002
                                 + 1.59847 w_k1_IBM_s002
                                 + 1.09734 w_k1_INTC_s002
                                 + 1.24547 w_k1_KO_s002 + 2.26662 w_k1_MAT_s002
                                 + 1.21464 w_k1_MSFT_s002
                                 + 1.19239 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.27167 w_k1_ADM_s002
                                 + 1.40812 w_k1_AXP_s002
                                 + 1.60067 w_k1_DIS_s002
                                 + 1.37412 w_k1_FDX_s002
                                 + 1.23144 w_k1_IBM_s002
                                 + 1.21574 w_k1_INTC_s002
                                 + 1.36294 w_k1_KO_s002 + 1.99501 w_k1_MAT_s002
                                 + 1.39727 w_k1_MSFT_s002 + 1.1964 w_k1_PG_s002
                                 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.03202 w_k1_ADM_s002
                                 + 0.980116 w_k1_AXP_s002
                                 + 1.25805 w_k1_DIS_s002
                                 + 1.04388 w_k1_FDX_s002 + 1.3412 w_k1_IBM_s002
                                 + 1.19261 w_k1_INTC_s002
                                 + 1.23324 w_k1_KO_s002 + 1.71768 w_k1_MAT_s002
                                 + 1.07542 w_k1_MSFT_s002
                                 + 1.01407 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.39302 w_k1_ADM_s002
                                 + 1.05125 w_k1_AXP_s002
                                 + 1.36773 w_k1_DIS_s002
                                 + 1.23482 w_k1_FDX_s002
                                 + 0.854472 w_k1_IBM_s002
                                 + 1.18297 w_k1_INTC_s002
                                 + 0.963823 w_k1_KO_s002
                                 + 0.985126 w_k1_MAT_s002
                                 + 1.22532 w_k1_MSFT_s002
                                 + 1.07513 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.03837 w_k1_ADM_s002
                                 + 0.790959 w_k1_AXP_s002
                                 + 0.804568 w_k1_DIS_s002
                                 + 0.944138 w_k1_FDX_s002
                                 + 0.667822 w_k1_IBM_s002
                                 + 0.863589 w_k1_INTC_s002
                                 + 0.701982 w_k1_KO_s002
                                 + 0.685191 w_k1_MAT_s002
                                 + 0.992623 w_k1_MSFT_s002
                                 + 0.867226 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.759916 w_k1_ADM_s003
                                 + 1.04703 w_k1_AXP_s003
                                 + 0.831094 w_k1_DIS_s003
                                 + 0.820786 w_k1_FDX_s003
                                 + 1.15628 w_k1_IBM_s003
                                 + 0.826279 w_k1_INTC_s003
                                 + 0.845203 w_k1_KO_s003
                                 + 1.47119 w_k1_MAT_s003
                                 + 0.711876 w_k1_MSFT_s003
                                 + 1.01773 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.750756 w_k1_ADM_s003
                                 + 0.683812 w_k1_AXP_s003
                                 + 0.74259 w_k1_DIS_s003
                                 + 0.712823 w_k1_FDX_s003
                                 + 1.09596 w_k1_IBM_s003
                                 + 0.912182 w_k1_INTC_s003
                                 + 0.882878 w_k1_KO_s003
                                 + 0.838258 w_k1_MAT_s003
                                 + 0.894235 w_k1_MSFT_s003
                                 + 0.938392 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.04091 w_k1_ADM_s003
                                 + 1.13133 w_k1_AXP_s003
                                 + 1.04624 w_k1_DIS_s003
                                 + 1.09205 w_k1_FDX_s003
                                 + 0.932701 w_k1_IBM_s003
                                 + 1.08551 w_k1_INTC_s003
                                 + 0.968366 w_k1_KO_s003
                                 + 1.03405 w_k1_MAT_s003
                                 + 0.962542 w_k1_MSFT_s003
                                 + 1.07965 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.754422 w_k1_ADM_s003
                                 + 0.827398 w_k1_AXP_s003
                                 + 0.898409 w_k1_DIS_s003
                                 + 0.993313 w_k1_FDX_s003
                                 + 0.617392 w_k1_IBM_s003
                                 + 1.00254 w_k1_INTC_s003
                                 + 0.732902 w_k1_KO_s003
                                 + 0.563356 w_k1_MAT_s003
                                 + 0.906939 w_k1_MSFT_s003
                                 + 0.842444 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.922741 w_k1_ADM_s003
                                 + 1.26605 w_k1_AXP_s003
                                 + 1.27113 w_k1_DIS_s003
                                 + 1.09796 w_k1_FDX_s003
                                 + 1.58155 w_k1_IBM_s003
                                 + 1.17989 w_k1_INTC_s003
                                 + 1.31006 w_k1_KO_s003 + 1.89871 w_k1_MAT_s003
                                 + 1.04908 w_k1_MSFT_s003
                                 + 1.12225 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.589793 w_k1_ADM_s003
                                 + 0.722523 w_k1_AXP_s003
                                 + 0.691807 w_k1_DIS_s003
                                 + 0.803783 w_k1_FDX_s003
                                 + 0.773054 w_k1_IBM_s003
                                 + 0.828614 w_k1_INTC_s003
                                 + 0.734565 w_k1_KO_s003
                                 + 0.769013 w_k1_MAT_s003
                                 + 0.781373 w_k1_MSFT_s003
                                 + 0.847375 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.09173 w_k1_ADM_s003
                                 + 1.31985 w_k1_AXP_s003 + 1.7497 w_k1_DIS_s003
                                 + 1.23337 w_k1_FDX_s003
                                 + 1.28315 w_k1_IBM_s003
                                 + 1.19726 w_k1_INTC_s003
                                 + 1.37006 w_k1_KO_s003 + 1.40148 w_k1_MAT_s003
                                 + 1.33489 w_k1_MSFT_s003
                                 + 1.28034 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.902327 w_k1_ADM_s003
                                 + 0.990441 w_k1_AXP_s003
                                 + 1.09482 w_k1_DIS_s003
                                 + 0.968341 w_k1_FDX_s003
                                 + 1.10422 w_k1_IBM_s003
                                 + 1.21183 w_k1_INTC_s003
                                 + 1.12204 w_k1_KO_s003 + 0.96167 w_k1_MAT_s003
                                 + 1.08494 w_k1_MSFT_s003
                                 + 1.17818 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.923464 w_k1_ADM_s004
                                 + 0.718696 w_k1_AXP_s004
                                 + 0.599977 w_k1_DIS_s004
                                 + 0.699912 w_k1_FDX_s004
                                 + 0.657135 w_k1_IBM_s004
                                 + 0.648221 w_k1_INTC_s004
                                 + 0.66931 w_k1_KO_s004
                                 + 0.581286 w_k1_MAT_s004
                                 + 0.704636 w_k1_MSFT_s004
                                 + 0.740437 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.11339 w_k1_ADM_s004
                                 + 1.23799 w_k1_AXP_s004
                                 + 1.34048 w_k1_DIS_s004
                                 + 1.25906 w_k1_FDX_s004
                                 + 1.03249 w_k1_IBM_s004
                                 + 1.0479 w_k1_INTC_s004 + 1.00112 w_k1_KO_s004
                                 + 1.03868 w_k1_MAT_s004
                                 + 1.34461 w_k1_MSFT_s004
                                 + 1.22555 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.2251 w_k1_ADM_s004
                                 + 1.1509 w_k1_AXP_s004 + 1.0244 w_k1_DIS_s004
                                 + 0.918535 w_k1_FDX_s004
                                 + 1.04148 w_k1_IBM_s004
                                 + 0.980617 w_k1_INTC_s004
                                 + 1.03178 w_k1_KO_s004 + 1.16471 w_k1_MAT_s004
                                 + 1.07963 w_k1_MSFT_s004
                                 + 1.04279 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.13634 w_k1_ADM_s004
                                 + 1.26215 w_k1_AXP_s004
                                 + 0.939969 w_k1_DIS_s004
                                 + 1.10681 w_k1_FDX_s004
                                 + 0.69534 w_k1_IBM_s004
                                 + 0.907919 w_k1_INTC_s004
                                 + 0.831943 w_k1_KO_s004
                                 + 0.573681 w_k1_MAT_s004
                                 + 0.913225 w_k1_MSFT_s004
                                 + 0.93077 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.86519 w_k1_ADM_s004
                                 + 0.911366 w_k1_AXP_s004
                                 + 0.721598 w_k1_DIS_s004
                                 + 0.795734 w_k1_FDX_s004
                                 + 1.03963 w_k1_IBM_s004
                                 + 0.74726 w_k1_INTC_s004
                                 + 0.831857 w_k1_KO_s004
                                 + 0.983228 w_k1_MAT_s004
                                 + 0.720975 w_k1_MSFT_s004
                                 + 0.848566 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.957002 w_k1_ADM_s004
                                 + 0.979713 w_k1_AXP_s004
                                 + 0.974745 w_k1_DIS_s004
                                 + 0.960888 w_k1_FDX_s004
                                 + 0.806781 w_k1_IBM_s004
                                 + 0.851043 w_k1_INTC_s004
                                 + 0.817937 w_k1_KO_s004
                                 + 0.732688 w_k1_MAT_s004
                                 + 0.915972 w_k1_MSFT_s004
                                 + 0.943323 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.21332 w_k1_ADM_s004
                                 + 1.1505 w_k1_AXP_s004 + 1.10789 w_k1_DIS_s004
                                 + 0.992887 w_k1_FDX_s004
                                 + 1.61068 w_k1_IBM_s004
                                 + 0.939539 w_k1_INTC_s004
                                 + 1.24886 w_k1_KO_s004 + 1.71692 w_k1_MAT_s004
                                 + 1.14374 w_k1_MSFT_s004
                                 + 0.985672 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.15673 w_k1_ADM_s004
                                 + 0.832654 w_k1_AXP_s004
                                 + 1.23414 w_k1_DIS_s004
                                 + 1.05535 w_k1_FDX_s004
                                 + 1.20912 w_k1_IBM_s004
                                 + 0.91606 w_k1_INTC_s004
                                 + 1.23978 w_k1_KO_s004
                                 + 0.921717 w_k1_MAT_s004
                                 + 1.09297 w_k1_MSFT_s004
                                 + 1.07182 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.02051 w_k1_ADM_s005
                                 + 1.19086 w_k1_AXP_s005
                                 + 1.03716 w_k1_DIS_s005
                                 + 0.980887 w_k1_FDX_s005
                                 + 1.39018 w_k1_IBM_s005
                                 + 0.790538 w_k1_INTC_s005
                                 + 1.22634 w_k1_KO_s005 + 1.65911 w_k1_MAT_s005
                                 + 0.979466 w_k1_MSFT_s005
                                 + 1.07711 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.989341 w_k1_ADM_s005
                                 + 1.19348 w_k1_AXP_s005
                                 + 1.03213 w_k1_DIS_s005
                                 + 1.15098 w_k1_FDX_s005
                                 + 0.815394 w_k1_IBM_s005
                                 + 0.890346 w_k1_INTC_s005
                                 + 0.921348 w_k1_KO_s005
                                 + 1.56901 w_k1_MAT_s005
                                 + 1.02658 w_k1_MSFT_s005
                                 + 0.953185 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.17037 w_k1_ADM_s005
                                 + 1.31905 w_k1_AXP_s005
                                 + 1.61041 w_k1_DIS_s005 + 1.4477 w_k1_FDX_s005
                                 + 0.934866 w_k1_IBM_s005
                                 + 1.24728 w_k1_INTC_s005
                                 + 1.03203 w_k1_KO_s005 + 1.11278 w_k1_MAT_s005
                                 + 1.36435 w_k1_MSFT_s005
                                 + 0.982622 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.05077 w_k1_ADM_s005
                                 + 1.4376 w_k1_AXP_s005 + 1.40585 w_k1_DIS_s005
                                 + 1.19773 w_k1_FDX_s005
                                 + 1.63818 w_k1_IBM_s005
                                 + 1.2978 w_k1_INTC_s005 + 1.49553 w_k1_KO_s005
                                 + 1.93443 w_k1_MAT_s005
                                 + 1.44538 w_k1_MSFT_s005
                                 + 1.15567 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.686992 w_k1_ADM_s005
                                 + 0.637712 w_k1_AXP_s005
                                 + 0.582777 w_k1_DIS_s005
                                 + 0.770379 w_k1_FDX_s005
                                 + 0.767767 w_k1_IBM_s005
                                 + 0.797338 w_k1_INTC_s005
                                 + 0.72174 w_k1_KO_s005
                                 + 0.760286 w_k1_MAT_s005
                                 + 0.730311 w_k1_MSFT_s005
                                 + 0.778513 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.780954 w_k1_ADM_s005
                                 + 0.732749 w_k1_AXP_s005
                                 + 0.952334 w_k1_DIS_s005
                                 + 0.96644 w_k1_FDX_s005
                                 + 0.885538 w_k1_IBM_s005
                                 + 0.933395 w_k1_INTC_s005
                                 + 0.962851 w_k1_KO_s005
                                 + 1.33173 w_k1_MAT_s005
                                 + 1.15039 w_k1_MSFT_s005
                                 + 0.88538 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.879319 w_k1_ADM_s005
                                 + 0.847438 w_k1_AXP_s005
                                 + 0.759565 w_k1_DIS_s005
                                 + 1.03136 w_k1_FDX_s005
                                 + 0.619357 w_k1_IBM_s005
                                 + 1.02013 w_k1_INTC_s005
                                 + 0.837173 w_k1_KO_s005
                                 + 0.575532 w_k1_MAT_s005
                                 + 1.04836 w_k1_MSFT_s005
                                 + 0.915151 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.07618 w_k1_ADM_s005
                                 + 0.821102 w_k1_AXP_s005
                                 + 0.858459 w_k1_DIS_s005
                                 + 0.885752 w_k1_FDX_s005
                                 + 0.816357 w_k1_IBM_s005
                                 + 0.828584 w_k1_INTC_s005
                                 + 0.841201 w_k1_KO_s005
                                 + 0.74077 w_k1_MAT_s005
                                 + 0.846267 w_k1_MSFT_s005
                                 + 0.866136 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.28787 w_k1_ADM_s006
                                 + 0.88659 w_k1_AXP_s006
                                 + 1.16999 w_k1_DIS_s006
                                 + 1.17669 w_k1_FDX_s006
                                 + 0.825776 w_k1_IBM_s006
                                 + 1.1404 w_k1_INTC_s006 + 1.08011 w_k1_KO_s006
                                 + 0.768247 w_k1_MAT_s006
                                 + 1.29115 w_k1_MSFT_s006
                                 + 1.08802 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.931375 w_k1_ADM_s006
                                 + 1.0166 w_k1_AXP_s006
                                 + 0.728662 w_k1_DIS_s006
                                 + 0.912401 w_k1_FDX_s006
                                 + 0.82823 w_k1_IBM_s006
                                 + 0.933888 w_k1_INTC_s006
                                 + 0.891906 w_k1_KO_s006
                                 + 1.18194 w_k1_MAT_s006
                                 + 0.965956 w_k1_MSFT_s006
                                 + 0.955249 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.16264 w_k1_ADM_s006
                                 + 1.22683 w_k1_AXP_s006
                                 + 1.02303 w_k1_DIS_s006
                                 + 1.00113 w_k1_FDX_s006
                                 + 1.55032 w_k1_IBM_s006
                                 + 1.09607 w_k1_INTC_s006
                                 + 1.14052 w_k1_KO_s006 + 1.85608 w_k1_MAT_s006
                                 + 0.982539 w_k1_MSFT_s006
                                 + 1.0081 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.828262 w_k1_ADM_s006
                                 + 0.870379 w_k1_AXP_s006
                                 + 1.02207 w_k1_DIS_s006
                                 + 1.01962 w_k1_FDX_s006
                                 + 0.731042 w_k1_IBM_s006
                                 + 1.02792 w_k1_INTC_s006
                                 + 1.05428 w_k1_KO_s006
                                 + 0.884293 w_k1_MAT_s006
                                 + 1.16672 w_k1_MSFT_s006
                                 + 0.985255 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.03583 w_k1_ADM_s006
                                 + 1.05787 w_k1_AXP_s006
                                 + 1.09781 w_k1_DIS_s006
                                 + 1.00932 w_k1_FDX_s006
                                 + 1.07473 w_k1_IBM_s006
                                 + 0.89323 w_k1_INTC_s006
                                 + 1.04507 w_k1_KO_s006 + 1.38321 w_k1_MAT_s006
                                 + 1.0341 w_k1_MSFT_s006
                                 + 0.965625 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.24083 w_k1_ADM_s006
                                 + 1.25927 w_k1_AXP_s006
                                 + 1.16837 w_k1_DIS_s006
                                 + 1.26718 w_k1_FDX_s006
                                 + 0.772636 w_k1_IBM_s006
                                 + 1.15622 w_k1_INTC_s006
                                 + 1.02065 w_k1_KO_s006
                                 + 0.689063 w_k1_MAT_s006
                                 + 1.14373 w_k1_MSFT_s006
                                 + 1.05651 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.844069 w_k1_ADM_s006
                                 + 0.70906 w_k1_AXP_s006
                                 + 0.67117 w_k1_DIS_s006
                                 + 0.722908 w_k1_FDX_s006
                                 + 0.743342 w_k1_IBM_s006
                                 + 0.790944 w_k1_INTC_s006
                                 + 0.859761 w_k1_KO_s006
                                 + 0.762124 w_k1_MAT_s006
                                 + 0.809578 w_k1_MSFT_s006
                                 + 0.936991 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.17144 w_k1_ADM_s006
                                 + 1.60208 w_k1_AXP_s006
                                 + 1.62894 w_k1_DIS_s006
                                 + 1.47278 w_k1_FDX_s006 + 1.0646 w_k1_IBM_s006
                                 + 1.4881 w_k1_INTC_s006 + 1.40144 w_k1_KO_s006
                                 + 1.31473 w_k1_MAT_s006
                                 + 1.48325 w_k1_MSFT_s006
                                 + 1.36193 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.672403 w_k1_ADM_s007
                                 + 0.623888 w_k1_AXP_s007
                                 + 0.624952 w_k1_DIS_s007
                                 + 0.702056 w_k1_FDX_s007
                                 + 0.859543 w_k1_IBM_s007
                                 + 0.658206 w_k1_INTC_s007
                                 + 0.754071 w_k1_KO_s007
                                 + 0.783682 w_k1_MAT_s007
                                 + 0.683346 w_k1_MSFT_s007
                                 + 0.832715 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.948696 w_k1_ADM_s007
                                 + 1.14753 w_k1_AXP_s007
                                 + 1.36541 w_k1_DIS_s007
                                 + 1.06289 w_k1_FDX_s007
                                 + 1.19179 w_k1_IBM_s007
                                 + 0.870716 w_k1_INTC_s007
                                 + 1.26163 w_k1_KO_s007 + 1.11645 w_k1_MAT_s007
                                 + 1.22958 w_k1_MSFT_s007
                                 + 1.13595 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.858163 w_k1_ADM_s007
                                 + 0.656824 w_k1_AXP_s007
                                 + 0.666688 w_k1_DIS_s007
                                 + 0.809255 w_k1_FDX_s007
                                 + 0.562098 w_k1_IBM_s007
                                 + 0.774727 w_k1_INTC_s007
                                 + 0.721576 w_k1_KO_s007
                                 + 0.638155 w_k1_MAT_s007
                                 + 0.996984 w_k1_MSFT_s007
                                 + 0.94855 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.24052 w_k1_ADM_s007
                                 + 1.09554 w_k1_AXP_s007
                                 + 1.20743 w_k1_DIS_s007
                                 + 1.17467 w_k1_FDX_s007
                                 + 0.81269 w_k1_IBM_s007
                                 + 0.882366 w_k1_INTC_s007
                                 + 0.958861 w_k1_KO_s007
                                 + 0.697741 w_k1_MAT_s007
                                 + 1.04093 w_k1_MSFT_s007 + 1.0922 w_k1_PG_s007
                                 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.18712 w_k1_ADM_s007
                                 + 1.44257 w_k1_AXP_s007 + 1.3817 w_k1_DIS_s007
                                 + 1.26866 w_k1_FDX_s007
                                 + 0.98092 w_k1_IBM_s007
                                 + 1.00169 w_k1_INTC_s007
                                 + 1.06296 w_k1_KO_s007 + 1.46228 w_k1_MAT_s007
                                 + 1.11495 w_k1_MSFT_s007
                                 + 1.21337 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.52983 w_k1_ADM_s007
                                 + 1.55153 w_k1_AXP_s007
                                 + 1.83512 w_k1_DIS_s007
                                 + 1.36986 w_k1_FDX_s007
                                 + 1.47983 w_k1_IBM_s007
                                 + 1.26604 w_k1_INTC_s007
                                 + 1.49396 w_k1_KO_s007 + 1.30404 w_k1_MAT_s007
                                 + 1.37839 w_k1_MSFT_s007
                                 + 1.48188 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.893373 w_k1_ADM_s007
                                 + 0.800117 w_k1_AXP_s007
                                 + 0.940193 w_k1_DIS_s007
                                 + 0.813959 w_k1_FDX_s007
                                 + 1.2568 w_k1_IBM_s007
                                 + 0.850262 w_k1_INTC_s007
                                 + 1.1039 w_k1_KO_s007 + 1.18292 w_k1_MAT_s007
                                 + 0.894443 w_k1_MSFT_s007
                                 + 1.07371 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.913015 w_k1_ADM_s007
                                 + 0.962406 w_k1_AXP_s007
                                 + 0.82483 w_k1_DIS_s007
                                 + 0.878755 w_k1_FDX_s007
                                 + 0.790902 w_k1_IBM_s007
                                 + 0.706959 w_k1_INTC_s007
                                 + 0.81379 w_k1_KO_s007 + 1.07799 w_k1_MAT_s007
                                 + 0.90622 w_k1_MSFT_s007
                                 + 0.921972 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.660865 w_k1_ADM_s010
                                 + 0.686589 w_k1_AXP_s010
                                 + 0.724629 w_k1_DIS_s010
                                 + 0.787267 w_k1_FDX_s010
                                 + 0.832952 w_k1_IBM_s010
                                 + 0.8993 w_k1_INTC_s010 + 0.72331 w_k1_KO_s010
                                 + 0.888635 w_k1_MAT_s010
                                 + 0.854269 w_k1_MSFT_s010
                                 + 0.799339 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.23184 w_k1_ADM_s010
                                 + 1.00491 w_k1_AXP_s010
                                 + 0.822377 w_k1_DIS_s010
                                 + 1.08024 w_k1_FDX_s010
                                 + 0.778078 w_k1_IBM_s010
                                 + 1.01401 w_k1_INTC_s010
                                 + 0.800618 w_k1_KO_s010
                                 + 0.820781 w_k1_MAT_s010
                                 + 0.895677 w_k1_MSFT_s010
                                 + 0.951503 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.59258 w_k1_ADM_s010
                                 + 1.6226 w_k1_AXP_s010 + 1.43129 w_k1_DIS_s010
                                 + 1.50972 w_k1_FDX_s010 + 1.4526 w_k1_IBM_s010
                                 + 1.3474 w_k1_INTC_s010 + 1.2243 w_k1_KO_s010
                                 + 1.38049 w_k1_MAT_s010
                                 + 1.16648 w_k1_MSFT_s010
                                 + 0.948787 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.971454 w_k1_ADM_s010
                                 + 1.09628 w_k1_AXP_s010
                                 + 1.44674 w_k1_DIS_s010
                                 + 1.29649 w_k1_FDX_s010
                                 + 1.19038 w_k1_IBM_s010
                                 + 0.954132 w_k1_INTC_s010
                                 + 0.953765 w_k1_KO_s010
                                 + 1.12978 w_k1_MAT_s010
                                 + 1.07456 w_k1_MSFT_s010
                                 + 0.952617 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.880322 w_k1_ADM_s010
                                 + 1.01834 w_k1_AXP_s010
                                 + 1.04731 w_k1_DIS_s010
                                 + 0.985894 w_k1_FDX_s010
                                 + 1.25393 w_k1_IBM_s010
                                 + 1.2004 w_k1_INTC_s010 + 1.1228 w_k1_KO_s010
                                 + 1.21301 w_k1_MAT_s010
                                 + 0.894981 w_k1_MSFT_s010
                                 + 1.02405 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.07064 w_k1_ADM_s010
                                 + 0.819707 w_k1_AXP_s010
                                 + 0.655692 w_k1_DIS_s010
                                 + 0.806901 w_k1_FDX_s010
                                 + 1.20826 w_k1_IBM_s010
                                 + 0.845095 w_k1_INTC_s010
                                 + 0.846749 w_k1_KO_s010
                                 + 1.32907 w_k1_MAT_s010
                                 + 0.810102 w_k1_MSFT_s010
                                 + 0.894512 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.94859 w_k1_ADM_s010
                                 + 0.780001 w_k1_AXP_s010
                                 + 0.814674 w_k1_DIS_s010
                                 + 0.923028 w_k1_FDX_s010
                                 + 0.648101 w_k1_IBM_s010
                                 + 1.20248 w_k1_INTC_s010
                                 + 0.778818 w_k1_KO_s010
                                 + 0.671272 w_k1_MAT_s010
                                 + 0.853271 w_k1_MSFT_s010
                                 + 0.822618 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.981415 w_k1_ADM_s010
                                 + 0.942932 w_k1_AXP_s010
                                 + 1.12861 w_k1_DIS_s010
                                 + 1.03585 w_k1_FDX_s010
                                 + 0.887306 w_k1_IBM_s010
                                 + 1.02874 w_k1_INTC_s010
                                 + 0.914037 w_k1_KO_s010
                                 + 0.748899 w_k1_MAT_s010
                                 + 1.08127 w_k1_MSFT_s010
                                 + 0.878173 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.650398 w_k1_ADM_s011
                                 + 0.697827 w_k1_AXP_s011
                                 + 0.702222 w_k1_DIS_s011
                                 + 0.714601 w_k1_FDX_s011
                                 + 0.938571 w_k1_IBM_s011
                                 + 0.821759 w_k1_INTC_s011
                                 + 0.823021 w_k1_KO_s011
                                 + 0.859518 w_k1_MAT_s011
                                 + 0.877243 w_k1_MSFT_s011
                                 + 0.847209 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.923758 w_k1_ADM_s011
                                 + 0.982374 w_k1_AXP_s011
                                 + 0.967869 w_k1_DIS_s011
                                 + 0.889584 w_k1_FDX_s011
                                 + 1.33838 w_k1_IBM_s011
                                 + 0.898459 w_k1_INTC_s011 + 1.185 w_k1_KO_s011
                                 + 1.02557 w_k1_MAT_s011
                                 + 1.09486 w_k1_MSFT_s011
                                 + 1.10089 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.726574 w_k1_ADM_s011
                                 + 0.7851 w_k1_AXP_s011
                                 + 0.895216 w_k1_DIS_s011
                                 + 0.913075 w_k1_FDX_s011
                                 + 1.02604 w_k1_IBM_s011
                                 + 0.993817 w_k1_INTC_s011
                                 + 0.903041 w_k1_KO_s011
                                 + 0.565554 w_k1_MAT_s011
                                 + 0.878323 w_k1_MSFT_s011
                                 + 0.899076 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.904233 w_k1_ADM_s011
                                 + 0.870403 w_k1_AXP_s011
                                 + 0.901289 w_k1_DIS_s011
                                 + 1.08104 w_k1_FDX_s011
                                 + 0.669965 w_k1_IBM_s011
                                 + 1.22688 w_k1_INTC_s011
                                 + 0.74234 w_k1_KO_s011
                                 + 0.500656 w_k1_MAT_s011
                                 + 1.15427 w_k1_MSFT_s011
                                 + 0.899563 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.95169 w_k1_ADM_s011
                                 + 1.14426 w_k1_AXP_s011
                                 + 1.17088 w_k1_DIS_s011
                                 + 1.13892 w_k1_FDX_s011
                                 + 0.946641 w_k1_IBM_s011
                                 + 1.01349 w_k1_INTC_s011
                                 + 1.03675 w_k1_KO_s011
                                 + 0.901594 w_k1_MAT_s011
                                 + 1.18217 w_k1_MSFT_s011
                                 + 0.983278 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.795711 w_k1_ADM_s011
                                 + 1.13669 w_k1_AXP_s011
                                 + 1.35814 w_k1_DIS_s011
                                 + 1.02078 w_k1_FDX_s011
                                 + 2.01063 w_k1_IBM_s011
                                 + 1.35403 w_k1_INTC_s011
                                 + 1.26288 w_k1_KO_s011 + 1.88696 w_k1_MAT_s011
                                 + 1.15614 w_k1_MSFT_s011
                                 + 1.14685 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.02778 w_k1_ADM_s011
                                 + 1.39645 w_k1_AXP_s011
                                 + 1.62781 w_k1_DIS_s011
                                 + 1.33067 w_k1_FDX_s011
                                 + 1.20045 w_k1_IBM_s011
                                 + 1.43816 w_k1_INTC_s011
                                 + 1.35617 w_k1_KO_s011
                                 + 0.848509 w_k1_MAT_s011
                                 + 1.37415 w_k1_MSFT_s011
                                 + 1.28059 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.942777 w_k1_ADM_s011
                                 + 1.20899 w_k1_AXP_s011
                                 + 1.39139 w_k1_DIS_s011
                                 + 1.05487 w_k1_FDX_s011
                                 + 1.42753 w_k1_IBM_s011
                                 + 1.01804 w_k1_INTC_s011
                                 + 1.38659 w_k1_KO_s011 + 1.3202 w_k1_MAT_s011
                                 + 1.15959 w_k1_MSFT_s011 + 1.2045 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.896732 w_k1_ADM_s012
                                 + 0.644353 w_k1_AXP_s012
                                 + 0.721456 w_k1_DIS_s012
                                 + 0.817766 w_k1_FDX_s012
                                 + 0.646066 w_k1_IBM_s012
                                 + 1.03612 w_k1_INTC_s012
                                 + 0.91351 w_k1_KO_s012
                                 + 0.626626 w_k1_MAT_s012
                                 + 0.92178 w_k1_MSFT_s012
                                 + 0.901982 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.19821 w_k1_ADM_s012
                                 + 0.802783 w_k1_AXP_s012
                                 + 1.12029 w_k1_DIS_s012
                                 + 0.977817 w_k1_FDX_s012
                                 + 1.04937 w_k1_IBM_s012
                                 + 1.08717 w_k1_INTC_s012
                                 + 1.19445 w_k1_KO_s012 + 0.79258 w_k1_MAT_s012
                                 + 0.99242 w_k1_MSFT_s012
                                 + 0.989203 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.32798 w_k1_ADM_s012
                                 + 1.12617 w_k1_AXP_s012
                                 + 1.02737 w_k1_DIS_s012
                                 + 1.00305 w_k1_FDX_s012
                                 + 1.23121 w_k1_IBM_s012
                                 + 1.00789 w_k1_INTC_s012
                                 + 1.17019 w_k1_KO_s012 + 1.07141 w_k1_MAT_s012
                                 + 0.969935 w_k1_MSFT_s012
                                 + 0.962936 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.976336 w_k1_ADM_s012
                                 + 0.933163 w_k1_AXP_s012
                                 + 0.916963 w_k1_DIS_s012
                                 + 0.998253 w_k1_FDX_s012
                                 + 0.85578 w_k1_IBM_s012
                                 + 1.33661 w_k1_INTC_s012
                                 + 1.13369 w_k1_KO_s012
                                 + 0.717213 w_k1_MAT_s012
                                 + 1.04097 w_k1_MSFT_s012
                                 + 0.933023 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.799746 w_k1_ADM_s012
                                 + 0.88468 w_k1_AXP_s012
                                 + 0.64416 w_k1_DIS_s012
                                 + 0.689309 w_k1_FDX_s012
                                 + 1.38964 w_k1_IBM_s012
                                 + 0.807144 w_k1_INTC_s012
                                 + 1.01346 w_k1_KO_s012 + 1.32405 w_k1_MAT_s012
                                 + 0.802647 w_k1_MSFT_s012
                                 + 0.878853 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.773923 w_k1_ADM_s012
                                 + 0.551567 w_k1_AXP_s012
                                 + 0.58063 w_k1_DIS_s012
                                 + 0.571575 w_k1_FDX_s012
                                 + 0.931344 w_k1_IBM_s012
                                 + 0.573916 w_k1_INTC_s012
                                 + 0.93394 w_k1_KO_s012
                                 + 0.736475 w_k1_MAT_s012
                                 + 0.65925 w_k1_MSFT_s012
                                 + 0.836622 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.04387 w_k1_ADM_s012
                                 + 0.781732 w_k1_AXP_s012
                                 + 0.87069 w_k1_DIS_s012
                                 + 0.832205 w_k1_FDX_s012
                                 + 0.723521 w_k1_IBM_s012
                                 + 0.955718 w_k1_INTC_s012
                                 + 0.945929 w_k1_KO_s012
                                 + 0.645664 w_k1_MAT_s012
                                 + 0.764002 w_k1_MSFT_s012
                                 + 0.911146 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.46489 w_k1_ADM_s012
                                 + 1.41169 w_k1_AXP_s012
                                 + 2.02889 w_k1_DIS_s012
                                 + 1.34362 w_k1_FDX_s012
                                 + 1.55776 w_k1_IBM_s012
                                 + 1.1971 w_k1_INTC_s012 + 1.84292 w_k1_KO_s012
                                 + 1.67604 w_k1_MAT_s012
                                 + 1.27924 w_k1_MSFT_s012
                                 + 1.51636 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.18887 w_k1_ADM_s013
                                 + 0.737407 w_k1_AXP_s013
                                 + 0.787718 w_k1_DIS_s013
                                 + 0.937878 w_k1_FDX_s013
                                 + 0.733865 w_k1_IBM_s013
                                 + 0.870517 w_k1_INTC_s013
                                 + 0.73174 w_k1_KO_s013
                                 + 0.721321 w_k1_MAT_s013
                                 + 0.900348 w_k1_MSFT_s013
                                 + 0.848463 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.47285 w_k1_ADM_s013
                                 + 1.25505 w_k1_AXP_s013
                                 + 1.01912 w_k1_DIS_s013 + 1.2625 w_k1_FDX_s013
                                 + 0.812565 w_k1_IBM_s013
                                 + 0.987997 w_k1_INTC_s013
                                 + 0.997691 w_k1_KO_s013
                                 + 1.27853 w_k1_MAT_s013
                                 + 1.0075 w_k1_MSFT_s013 + 1.08872 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.04765 w_k1_ADM_s013
                                 + 1.01758 w_k1_AXP_s013
                                 + 1.24856 w_k1_DIS_s013 + 1.1648 w_k1_FDX_s013
                                 + 1.45253 w_k1_IBM_s013
                                 + 1.13674 w_k1_INTC_s013
                                 + 1.21363 w_k1_KO_s013 + 1.40299 w_k1_MAT_s013
                                 + 1.09617 w_k1_MSFT_s013
                                 + 1.03795 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.694355 w_k1_ADM_s013
                                 + 0.622411 w_k1_AXP_s013
                                 + 0.674666 w_k1_DIS_s013
                                 + 0.780649 w_k1_FDX_s013
                                 + 1.16686 w_k1_IBM_s013
                                 + 1.04312 w_k1_INTC_s013
                                 + 0.922127 w_k1_KO_s013
                                 + 0.896633 w_k1_MAT_s013
                                 + 0.872835 w_k1_MSFT_s013
                                 + 0.876776 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.10438 w_k1_ADM_s013
                                 + 0.927767 w_k1_AXP_s013
                                 + 1.31294 w_k1_DIS_s013
                                 + 1.12516 w_k1_FDX_s013
                                 + 0.911295 w_k1_IBM_s013
                                 + 1.15156 w_k1_INTC_s013
                                 + 1.14098 w_k1_KO_s013 + 1.02688 w_k1_MAT_s013
                                 + 1.20887 w_k1_MSFT_s013
                                 + 1.09215 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.677426 w_k1_ADM_s013
                                 + 0.742432 w_k1_AXP_s013
                                 + 0.615177 w_k1_DIS_s013
                                 + 0.822043 w_k1_FDX_s013
                                 + 0.652362 w_k1_IBM_s013
                                 + 1.04642 w_k1_INTC_s013
                                 + 0.728097 w_k1_KO_s013
                                 + 0.587611 w_k1_MAT_s013
                                 + 0.801372 w_k1_MSFT_s013
                                 + 0.741815 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.08877 w_k1_ADM_s013
                                 + 1.34759 w_k1_AXP_s013
                                 + 1.60865 w_k1_DIS_s013
                                 + 1.49844 w_k1_FDX_s013
                                 + 1.07906 w_k1_IBM_s013
                                 + 1.41364 w_k1_INTC_s013
                                 + 1.19812 w_k1_KO_s013 + 1.28823 w_k1_MAT_s013
                                 + 1.37016 w_k1_MSFT_s013
                                 + 1.21164 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.831399 w_k1_ADM_s013
                                 + 0.961054 w_k1_AXP_s013
                                 + 0.942465 w_k1_DIS_s013
                                 + 0.987535 w_k1_FDX_s013
                                 + 1.06304 w_k1_IBM_s013
                                 + 1.04047 w_k1_INTC_s013
                                 + 0.928601 w_k1_KO_s013
                                 + 1.27872 w_k1_MAT_s013
                                 + 1.01899 w_k1_MSFT_s013
                                 + 0.93707 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.03424 w_k1_ADM_s014
                                 + 0.845268 w_k1_AXP_s014
                                 + 0.760011 w_k1_DIS_s014
                                 + 0.830201 w_k1_FDX_s014
                                 + 0.833576 w_k1_IBM_s014
                                 + 0.887341 w_k1_INTC_s014
                                 + 0.830931 w_k1_KO_s014
                                 + 0.722944 w_k1_MAT_s014
                                 + 0.838958 w_k1_MSFT_s014
                                 + 1.0017 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.29673 w_k1_ADM_s014
                                 + 1.28695 w_k1_AXP_s014
                                 + 1.66268 w_k1_DIS_s014
                                 + 1.50864 w_k1_FDX_s014 + 1.1508 w_k1_IBM_s014
                                 + 1.37759 w_k1_INTC_s014
                                 + 1.30281 w_k1_KO_s014
                                 + 0.967913 w_k1_MAT_s014
                                 + 1.24156 w_k1_MSFT_s014
                                 + 1.12753 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.03988 w_k1_ADM_s014
                                 + 0.816365 w_k1_AXP_s014
                                 + 0.73807 w_k1_DIS_s014
                                 + 0.839472 w_k1_FDX_s014
                                 + 1.1999 w_k1_IBM_s014 + 1.0631 w_k1_INTC_s014
                                 + 0.88684 w_k1_KO_s014 + 1.00999 w_k1_MAT_s014
                                 + 0.861694 w_k1_MSFT_s014
                                 + 0.930178 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.05389 w_k1_ADM_s014
                                 + 1.28348 w_k1_AXP_s014
                                 + 1.12011 w_k1_DIS_s014
                                 + 1.12303 w_k1_FDX_s014
                                 + 0.997926 w_k1_IBM_s014
                                 + 1.23902 w_k1_INTC_s014
                                 + 0.95853 w_k1_KO_s014
                                 + 0.995548 w_k1_MAT_s014
                                 + 1.08301 w_k1_MSFT_s014
                                 + 1.00296 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.09124 w_k1_ADM_s014
                                 + 0.98226 w_k1_AXP_s014
                                 + 1.12898 w_k1_DIS_s014
                                 + 0.925387 w_k1_FDX_s014
                                 + 1.2362 w_k1_IBM_s014
                                 + 1.12482 w_k1_INTC_s014
                                 + 1.07684 w_k1_KO_s014
                                 + 0.989867 w_k1_MAT_s014
                                 + 1.04204 w_k1_MSFT_s014
                                 + 0.963659 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.1355 w_k1_ADM_s014
                                 + 1.09579 w_k1_AXP_s014 + 1.0296 w_k1_DIS_s014
                                 + 1.17486 w_k1_FDX_s014
                                 + 0.736669 w_k1_IBM_s014
                                 + 1.04522 w_k1_INTC_s014
                                 + 0.918007 w_k1_KO_s014
                                 + 0.917635 w_k1_MAT_s014
                                 + 1.06078 w_k1_MSFT_s014
                                 + 0.991762 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.85545 w_k1_ADM_s014
                                 + 1.42514 w_k1_AXP_s014
                                 + 1.22523 w_k1_DIS_s014 + 1.0604 w_k1_FDX_s014
                                 + 1.57595 w_k1_IBM_s014
                                 + 0.882084 w_k1_INTC_s014
                                 + 1.20886 w_k1_KO_s014 + 1.86205 w_k1_MAT_s014
                                 + 1.02253 w_k1_MSFT_s014
                                 + 1.21159 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.966012 w_k1_ADM_s014
                                 + 1.06512 w_k1_AXP_s014
                                 + 0.958056 w_k1_DIS_s014
                                 + 0.896584 w_k1_FDX_s014
                                 + 1.5094 w_k1_IBM_s014
                                 + 0.908173 w_k1_INTC_s014
                                 + 1.18088 w_k1_KO_s014 + 1.94868 w_k1_MAT_s014
                                 + 1.0321 w_k1_MSFT_s014 + 1.054 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.01845 w_k1_ADM_s015
                                 + 1.09158 w_k1_AXP_s015
                                 + 1.28533 w_k1_DIS_s015
                                 + 1.03807 w_k1_FDX_s015
                                 + 1.53319 w_k1_IBM_s015
                                 + 1.08683 w_k1_INTC_s015
                                 + 1.09418 w_k1_KO_s015 + 1.48896 w_k1_MAT_s015
                                 + 1.11729 w_k1_MSFT_s015
                                 + 0.966997 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.867287 w_k1_ADM_s015
                                 + 0.790095 w_k1_AXP_s015
                                 + 0.705553 w_k1_DIS_s015
                                 + 0.804026 w_k1_FDX_s015
                                 + 1.0717 w_k1_IBM_s015
                                 + 0.941045 w_k1_INTC_s015
                                 + 0.85285 w_k1_KO_s015
                                 + 0.815492 w_k1_MAT_s015
                                 + 0.935077 w_k1_MSFT_s015
                                 + 0.854154 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.25689 w_k1_ADM_s015
                                 + 1.07205 w_k1_AXP_s015
                                 + 1.10134 w_k1_DIS_s015
                                 + 1.16737 w_k1_FDX_s015
                                 + 0.780578 w_k1_IBM_s015
                                 + 0.918439 w_k1_INTC_s015
                                 + 0.796498 w_k1_KO_s015
                                 + 0.997532 w_k1_MAT_s015
                                 + 0.976799 w_k1_MSFT_s015
                                 + 0.90028 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.09854 w_k1_ADM_s015
                                 + 0.788922 w_k1_AXP_s015
                                 + 0.759307 w_k1_DIS_s015
                                 + 0.783817 w_k1_FDX_s015
                                 + 0.647877 w_k1_IBM_s015
                                 + 0.745213 w_k1_INTC_s015
                                 + 0.704337 w_k1_KO_s015
                                 + 0.555327 w_k1_MAT_s015
                                 + 0.768419 w_k1_MSFT_s015
                                 + 0.780467 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.19438 w_k1_ADM_s015
                                 + 0.97217 w_k1_AXP_s015
                                 + 0.937659 w_k1_DIS_s015
                                 + 0.828637 w_k1_FDX_s015
                                 + 1.43137 w_k1_IBM_s015
                                 + 0.753839 w_k1_INTC_s015
                                 + 1.16773 w_k1_KO_s015 + 1.18204 w_k1_MAT_s015
                                 + 0.776691 w_k1_MSFT_s015
                                 + 1.04545 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.13045 w_k1_ADM_s015
                                 + 0.934748 w_k1_AXP_s015
                                 + 1.06808 w_k1_DIS_s015
                                 + 1.01075 w_k1_FDX_s015 + 1.2648 w_k1_IBM_s015
                                 + 1.02977 w_k1_INTC_s015
                                 + 0.946056 w_k1_KO_s015
                                 + 0.955169 w_k1_MAT_s015
                                 + 0.958046 w_k1_MSFT_s015
                                 + 0.987618 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.35139 w_k1_ADM_s015
                                 + 1.04203 w_k1_AXP_s015
                                 + 1.51926 w_k1_DIS_s015
                                 + 1.11792 w_k1_FDX_s015
                                 + 1.12539 w_k1_IBM_s015
                                 + 0.995526 w_k1_INTC_s015
                                 + 1.16143 w_k1_KO_s015
                                 + 0.971708 w_k1_MAT_s015
                                 + 1.11864 w_k1_MSFT_s015
                                 + 1.07415 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.23643 w_k1_ADM_s015
                                 + 1.31615 w_k1_AXP_s015
                                 + 1.64617 w_k1_DIS_s015 + 1.3165 w_k1_FDX_s015
                                 + 1.14076 w_k1_IBM_s015
                                 + 1.11139 w_k1_INTC_s015
                                 + 1.15442 w_k1_KO_s015 + 1.36742 w_k1_MAT_s015
                                 + 1.22979 w_k1_MSFT_s015
                                 + 1.08652 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.733497 w_k1_ADM_s016
                                 + 0.594229 w_k1_AXP_s016
                                 + 0.57307 w_k1_DIS_s016
                                 + 0.68007 w_k1_FDX_s016
                                 + 0.937419 w_k1_IBM_s016
                                 + 0.798412 w_k1_INTC_s016
                                 + 0.867084 w_k1_KO_s016
                                 + 0.57006 w_k1_MAT_s016
                                 + 0.749359 w_k1_MSFT_s016
                                 + 0.78845 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.82761 w_k1_ADM_s016
                                 + 0.99745 w_k1_AXP_s016
                                 + 0.835946 w_k1_DIS_s016
                                 + 0.813406 w_k1_FDX_s016
                                 + 1.48629 w_k1_IBM_s016
                                 + 1.18526 w_k1_INTC_s016
                                 + 1.02828 w_k1_KO_s016 + 1.25757 w_k1_MAT_s016
                                 + 0.995197 w_k1_MSFT_s016
                                 + 0.995702 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.1872 w_k1_ADM_s016
                                 + 1.11389 w_k1_AXP_s016
                                 + 0.991265 w_k1_DIS_s016
                                 + 0.979699 w_k1_FDX_s016
                                 + 1.11644 w_k1_IBM_s016
                                 + 1.12787 w_k1_INTC_s016
                                 + 1.12949 w_k1_KO_s016
                                 + 0.929974 w_k1_MAT_s016
                                 + 1.02406 w_k1_MSFT_s016
                                 + 1.04882 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.678749 w_k1_ADM_s016
                                 + 0.68977 w_k1_AXP_s016
                                 + 0.784233 w_k1_DIS_s016
                                 + 0.774197 w_k1_FDX_s016
                                 + 0.905578 w_k1_IBM_s016
                                 + 1.02329 w_k1_INTC_s016 + 1.107 w_k1_KO_s016
                                 + 0.99927 w_k1_MAT_s016
                                 + 1.04342 w_k1_MSFT_s016
                                 + 0.926398 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.22127 w_k1_ADM_s016
                                 + 1.33412 w_k1_AXP_s016
                                 + 1.29191 w_k1_DIS_s016
                                 + 1.29798 w_k1_FDX_s016
                                 + 1.09179 w_k1_IBM_s016
                                 + 1.22516 w_k1_INTC_s016
                                 + 1.19762 w_k1_KO_s016 + 1.16528 w_k1_MAT_s016
                                 + 1.21069 w_k1_MSFT_s016
                                 + 1.10276 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.09271 w_k1_ADM_s016
                                 + 1.37446 w_k1_AXP_s016
                                 + 1.56172 w_k1_DIS_s016
                                 + 1.13918 w_k1_FDX_s016
                                 + 1.63652 w_k1_IBM_s016
                                 + 1.09799 w_k1_INTC_s016
                                 + 1.57875 w_k1_KO_s016 + 1.35994 w_k1_MAT_s016
                                 + 1.25741 w_k1_MSFT_s016
                                 + 1.26603 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.14894 w_k1_ADM_s016
                                 + 1.04919 w_k1_AXP_s016 + 0.9011 w_k1_DIS_s016
                                 + 1.00693 w_k1_FDX_s016
                                 + 0.77016 w_k1_IBM_s016
                                 + 0.984682 w_k1_INTC_s016
                                 + 0.89894 w_k1_KO_s016
                                 + 0.782175 w_k1_MAT_s016
                                 + 0.94512 w_k1_MSFT_s016
                                 + 0.937842 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.766236 w_k1_ADM_s016
                                 + 0.677504 w_k1_AXP_s016
                                 + 0.651608 w_k1_DIS_s016
                                 + 0.759276 w_k1_FDX_s016
                                 + 0.55958 w_k1_IBM_s016
                                 + 0.7854 w_k1_INTC_s016
                                 + 0.689347 w_k1_KO_s016
                                 + 0.549075 w_k1_MAT_s016
                                 + 0.824612 w_k1_MSFT_s016
                                 + 0.788697 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.865914 w_k1_ADM_s017
                                 + 0.722886 w_k1_AXP_s017
                                 + 0.898595 w_k1_DIS_s017
                                 + 0.767653 w_k1_FDX_s017
                                 + 0.767116 w_k1_IBM_s017
                                 + 0.783324 w_k1_INTC_s017
                                 + 0.80059 w_k1_KO_s017
                                 + 0.782929 w_k1_MAT_s017
                                 + 0.887531 w_k1_MSFT_s017
                                 + 0.74383 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.00701 w_k1_ADM_s017
                                 + 0.945119 w_k1_AXP_s017
                                 + 0.870893 w_k1_DIS_s017
                                 + 0.992984 w_k1_FDX_s017
                                 + 0.759892 w_k1_IBM_s017
                                 + 0.984845 w_k1_INTC_s017
                                 + 0.901101 w_k1_KO_s017
                                 + 0.750681 w_k1_MAT_s017
                                 + 1.01185 w_k1_MSFT_s017
                                 + 0.758673 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.921232 w_k1_ADM_s017
                                 + 1.37539 w_k1_AXP_s017
                                 + 1.40837 w_k1_DIS_s017
                                 + 1.22668 w_k1_FDX_s017
                                 + 1.05935 w_k1_IBM_s017
                                 + 1.24293 w_k1_INTC_s017
                                 + 1.23017 w_k1_KO_s017 + 1.23948 w_k1_MAT_s017
                                 + 1.33191 w_k1_MSFT_s017
                                 + 0.906104 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.25218 w_k1_ADM_s017
                                 + 1.07758 w_k1_AXP_s017
                                 + 1.44513 w_k1_DIS_s017
                                 + 1.15333 w_k1_FDX_s017
                                 + 0.901866 w_k1_IBM_s017
                                 + 0.961266 w_k1_INTC_s017
                                 + 1.27597 w_k1_KO_s017 + 1.23036 w_k1_MAT_s017
                                 + 1.27393 w_k1_MSFT_s017
                                 + 0.975803 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.02986 w_k1_ADM_s017
                                 + 0.850314 w_k1_AXP_s017
                                 + 0.946181 w_k1_DIS_s017
                                 + 0.99077 w_k1_FDX_s017
                                 + 0.610278 w_k1_IBM_s017
                                 + 0.840539 w_k1_INTC_s017
                                 + 0.858329 w_k1_KO_s017
                                 + 0.508387 w_k1_MAT_s017
                                 + 1.04015 w_k1_MSFT_s017
                                 + 0.748363 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.13795 w_k1_ADM_s017
                                 + 1.22953 w_k1_AXP_s017
                                 + 1.00235 w_k1_DIS_s017
                                 + 0.972907 w_k1_FDX_s017
                                 + 1.39685 w_k1_IBM_s017
                                 + 0.922117 w_k1_INTC_s017
                                 + 1.11716 w_k1_KO_s017 + 1.50179 w_k1_MAT_s017
                                 + 0.938799 w_k1_MSFT_s017
                                 + 1.0321 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.935917 w_k1_ADM_s017
                                 + 0.910318 w_k1_AXP_s017
                                 + 0.588067 w_k1_DIS_s017
                                 + 0.799811 w_k1_FDX_s017
                                 + 0.838985 w_k1_IBM_s017
                                 + 0.982158 w_k1_INTC_s017
                                 + 0.750496 w_k1_KO_s017
                                 + 0.914904 w_k1_MAT_s017
                                 + 0.85273 w_k1_MSFT_s017
                                 + 0.765106 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.11469 w_k1_ADM_s017
                                 + 0.805528 w_k1_AXP_s017
                                 + 0.85842 w_k1_DIS_s017
                                 + 0.882617 w_k1_FDX_s017
                                 + 1.01801 w_k1_IBM_s017
                                 + 0.930657 w_k1_INTC_s017
                                 + 1.04622 w_k1_KO_s017 + 1.24314 w_k1_MAT_s017
                                 + 1.01898 w_k1_MSFT_s017
                                 + 0.926352 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.706777 w_k1_ADM_s100
                                 + 0.74412 w_k1_AXP_s100
                                 + 0.678517 w_k1_DIS_s100
                                 + 0.732365 w_k1_FDX_s100
                                 + 1.12126 w_k1_IBM_s100
                                 + 0.80501 w_k1_INTC_s100
                                 + 0.967063 w_k1_KO_s100
                                 + 0.878184 w_k1_MAT_s100
                                 + 0.847383 w_k1_MSFT_s100
                                 + 0.964295 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.13089 w_k1_ADM_s100
                                 + 1.01807 w_k1_AXP_s100
                                 + 1.19701 w_k1_DIS_s100
                                 + 1.25532 w_k1_FDX_s100
                                 + 0.946894 w_k1_IBM_s100
                                 + 1.14735 w_k1_INTC_s100
                                 + 1.09594 w_k1_KO_s100
                                 + 0.841551 w_k1_MAT_s100
                                 + 1.27099 w_k1_MSFT_s100 + 1.136 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.22462 w_k1_ADM_s100
                                 + 1.41816 w_k1_AXP_s100
                                 + 1.83409 w_k1_DIS_s100
                                 + 1.57345 w_k1_FDX_s100
                                 + 1.19937 w_k1_IBM_s100
                                 + 1.47015 w_k1_INTC_s100
                                 + 1.32611 w_k1_KO_s100 + 1.37365 w_k1_MAT_s100
                                 + 1.60062 w_k1_MSFT_s100
                                 + 1.26268 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.00312 w_k1_ADM_s100
                                 + 0.919291 w_k1_AXP_s100
                                 + 0.878431 w_k1_DIS_s100
                                 + 0.96425 w_k1_FDX_s100
                                 + 0.914026 w_k1_IBM_s100
                                 + 0.904036 w_k1_INTC_s100
                                 + 0.89853 w_k1_KO_s100 + 1.14479 w_k1_MAT_s100
                                 + 1.01062 w_k1_MSFT_s100
                                 + 1.02811 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.24365 w_k1_ADM_s100
                                 + 1.35822 w_k1_AXP_s100
                                 + 1.21117 w_k1_DIS_s100
                                 + 1.27762 w_k1_FDX_s100
                                 + 1.40802 w_k1_IBM_s100
                                 + 1.11156 w_k1_INTC_s100
                                 + 1.24495 w_k1_KO_s100 + 1.34441 w_k1_MAT_s100
                                 + 1.16168 w_k1_MSFT_s100
                                 + 1.07915 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.680321 w_k1_ADM_s100
                                 + 0.515764 w_k1_AXP_s100
                                 + 0.425178 w_k1_DIS_s100
                                 + 0.627365 w_k1_FDX_s100
                                 + 0.690748 w_k1_IBM_s100
                                 + 0.748334 w_k1_INTC_s100
                                 + 0.638091 w_k1_KO_s100
                                 + 0.603398 w_k1_MAT_s100
                                 + 0.685311 w_k1_MSFT_s100
                                 + 0.726276 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.95802 w_k1_ADM_s100
                                 + 0.767737 w_k1_AXP_s100
                                 + 0.639207 w_k1_DIS_s100
                                 + 0.798001 w_k1_FDX_s100
                                 + 0.884883 w_k1_IBM_s100
                                 + 0.810144 w_k1_INTC_s100
                                 + 0.867812 w_k1_KO_s100
                                 + 0.956168 w_k1_MAT_s100
                                 + 0.857214 w_k1_MSFT_s100
                                 + 0.866416 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.841073 w_k1_ADM_s100
                                 + 1.04501 w_k1_AXP_s100
                                 + 1.24359 w_k1_DIS_s100
                                 + 0.979231 w_k1_FDX_s100
                                 + 1.60992 w_k1_IBM_s100
                                 + 1.10416 w_k1_INTC_s100
                                 + 1.33344 w_k1_KO_s100 + 1.60029 w_k1_MAT_s100
                                 + 1.03344 w_k1_MSFT_s100
                                 + 1.12491 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.745364 w_k1_ADM_s101
                                 + 0.591557 w_k1_AXP_s101
                                 + 0.784285 w_k1_DIS_s101
                                 + 0.715934 w_k1_FDX_s101
                                 + 0.909129 w_k1_IBM_s101
                                 + 0.761444 w_k1_INTC_s101
                                 + 0.83065 w_k1_KO_s101
                                 + 0.910661 w_k1_MAT_s101
                                 + 0.853005 w_k1_MSFT_s101
                                 + 0.853521 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.16749 w_k1_ADM_s101
                                 + 1.14906 w_k1_AXP_s101
                                 + 1.30154 w_k1_DIS_s101
                                 + 1.10344 w_k1_FDX_s101
                                 + 1.35868 w_k1_IBM_s101
                                 + 1.10907 w_k1_INTC_s101
                                 + 1.28285 w_k1_KO_s101 + 1.34911 w_k1_MAT_s101
                                 + 1.04861 w_k1_MSFT_s101 + 1.1196 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.29642 w_k1_ADM_s101
                                 + 1.35616 w_k1_AXP_s101
                                 + 1.75195 w_k1_DIS_s101
                                 + 1.35484 w_k1_FDX_s101
                                 + 1.15568 w_k1_IBM_s101
                                 + 1.32313 w_k1_INTC_s101 + 1.3991 w_k1_KO_s101
                                 + 1.21385 w_k1_MAT_s101
                                 + 1.28337 w_k1_MSFT_s101
                                 + 1.15577 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.10038 w_k1_ADM_s101
                                 + 0.890232 w_k1_AXP_s101
                                 + 1.21458 w_k1_DIS_s101
                                 + 1.03237 w_k1_FDX_s101
                                 + 0.974547 w_k1_IBM_s101
                                 + 0.957654 w_k1_INTC_s101
                                 + 1.19711 w_k1_KO_s101
                                 + 0.881899 w_k1_MAT_s101
                                 + 1.0109 w_k1_MSFT_s101 + 1.04318 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.784407 w_k1_ADM_s101
                                 + 0.77711 w_k1_AXP_s101
                                 + 0.680038 w_k1_DIS_s101
                                 + 0.766138 w_k1_FDX_s101
                                 + 0.807302 w_k1_IBM_s101
                                 + 0.968697 w_k1_INTC_s101
                                 + 0.797881 w_k1_KO_s101
                                 + 0.624329 w_k1_MAT_s101
                                 + 0.81537 w_k1_MSFT_s101
                                 + 0.868199 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.05228 w_k1_ADM_s101
                                 + 0.952001 w_k1_AXP_s101
                                 + 0.919165 w_k1_DIS_s101
                                 + 0.989585 w_k1_FDX_s101
                                 + 0.819217 w_k1_IBM_s101
                                 + 0.983283 w_k1_INTC_s101
                                 + 0.808666 w_k1_KO_s101
                                 + 0.878021 w_k1_MAT_s101
                                 + 0.856213 w_k1_MSFT_s101
                                 + 0.889052 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.20092 w_k1_ADM_s101
                                 + 1.2056 w_k1_AXP_s101 + 1.0364 w_k1_DIS_s101
                                 + 1.19841 w_k1_FDX_s101
                                 + 0.625327 w_k1_IBM_s101
                                 + 1.05018 w_k1_INTC_s101
                                 + 0.801297 w_k1_KO_s101
                                 + 0.561657 w_k1_MAT_s101
                                 + 1.00315 w_k1_MSFT_s101
                                 + 0.996407 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.701835 w_k1_ADM_s101
                                 + 0.806376 w_k1_AXP_s101
                                 + 1.02604 w_k1_DIS_s101
                                 + 0.868065 w_k1_FDX_s101
                                 + 1.30233 w_k1_IBM_s101
                                 + 1.0845 w_k1_INTC_s101 + 1.27124 w_k1_KO_s101
                                 + 0.923941 w_k1_MAT_s101
                                 + 1.07444 w_k1_MSFT_s101
                                 + 1.05847 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.711482 w_k1_ADM_s102
                                 + 0.756775 w_k1_AXP_s102
                                 + 0.735501 w_k1_DIS_s102
                                 + 0.818153 w_k1_FDX_s102
                                 + 0.891129 w_k1_IBM_s102
                                 + 1.22981 w_k1_INTC_s102
                                 + 0.924961 w_k1_KO_s102
                                 + 0.772603 w_k1_MAT_s102
                                 + 0.847123 w_k1_MSFT_s102
                                 + 0.85493 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.958837 w_k1_ADM_s102
                                 + 0.915291 w_k1_AXP_s102
                                 + 1.05307 w_k1_DIS_s102 + 1.1757 w_k1_FDX_s102
                                 + 0.854536 w_k1_IBM_s102
                                 + 1.14531 w_k1_INTC_s102
                                 + 1.06239 w_k1_KO_s102
                                 + 0.980796 w_k1_MAT_s102
                                 + 1.06298 w_k1_MSFT_s102
                                 + 0.887285 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.89284 w_k1_ADM_s102
                                 + 1.14698 w_k1_AXP_s102
                                 + 0.945968 w_k1_DIS_s102
                                 + 0.907641 w_k1_FDX_s102
                                 + 1.35873 w_k1_IBM_s102
                                 + 1.14244 w_k1_INTC_s102 + 1.2727 w_k1_KO_s102
                                 + 1.26449 w_k1_MAT_s102
                                 + 1.10535 w_k1_MSFT_s102
                                 + 0.949411 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.17035 w_k1_ADM_s102
                                 + 1.45599 w_k1_AXP_s102
                                 + 1.61365 w_k1_DIS_s102
                                 + 1.29013 w_k1_FDX_s102
                                 + 1.35912 w_k1_IBM_s102
                                 + 1.41318 w_k1_INTC_s102
                                 + 1.43437 w_k1_KO_s102 + 1.44908 w_k1_MAT_s102
                                 + 1.50924 w_k1_MSFT_s102
                                 + 1.17749 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.923326 w_k1_ADM_s102
                                 + 1.10699 w_k1_AXP_s102
                                 + 1.09503 w_k1_DIS_s102
                                 + 0.996404 w_k1_FDX_s102
                                 + 1.09134 w_k1_IBM_s102
                                 + 1.29979 w_k1_INTC_s102
                                 + 1.06661 w_k1_KO_s102
                                 + 0.887461 w_k1_MAT_s102
                                 + 1.0339 w_k1_MSFT_s102
                                 + 0.970784 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.16674 w_k1_ADM_s102
                                 + 1.15498 w_k1_AXP_s102
                                 + 0.914558 w_k1_DIS_s102
                                 + 1.19506 w_k1_FDX_s102
                                 + 0.760803 w_k1_IBM_s102
                                 + 1.05746 w_k1_INTC_s102
                                 + 0.81158 w_k1_KO_s102
                                 + 0.788056 w_k1_MAT_s102
                                 + 1.01531 w_k1_MSFT_s102
                                 + 0.896021 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.18195 w_k1_ADM_s102
                                 + 1.25177 w_k1_AXP_s102
                                 + 1.30425 w_k1_DIS_s102
                                 + 1.24385 w_k1_FDX_s102
                                 + 1.13199 w_k1_IBM_s102
                                 + 1.27571 w_k1_INTC_s102
                                 + 1.25869 w_k1_KO_s102 + 1.20518 w_k1_MAT_s102
                                 + 1.24993 w_k1_MSFT_s102
                                 + 0.997931 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.05442 w_k1_ADM_s102
                                 + 1.01428 w_k1_AXP_s102
                                 + 1.34422 w_k1_DIS_s102
                                 + 1.11921 w_k1_FDX_s102
                                 + 0.939795 w_k1_IBM_s102
                                 + 1.41227 w_k1_INTC_s102
                                 + 1.14935 w_k1_KO_s102
                                 + 0.643536 w_k1_MAT_s102
                                 + 1.24766 w_k1_MSFT_s102
                                 + 1.03953 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.15599 w_k1_ADM_s103
                                 + 1.29558 w_k1_AXP_s103
                                 + 1.45632 w_k1_DIS_s103
                                 + 1.27299 w_k1_FDX_s103
                                 + 0.997369 w_k1_IBM_s103
                                 + 1.18379 w_k1_INTC_s103
                                 + 1.09562 w_k1_KO_s103 + 1.10586 w_k1_MAT_s103
                                 + 1.20503 w_k1_MSFT_s103
                                 + 1.13334 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.20001 w_k1_ADM_s103
                                 + 1.39746 w_k1_AXP_s103
                                 + 1.42186 w_k1_DIS_s103
                                 + 1.18811 w_k1_FDX_s103
                                 + 1.31846 w_k1_IBM_s103
                                 + 1.19864 w_k1_INTC_s103 + 1.0875 w_k1_KO_s103
                                 + 1.8964 w_k1_MAT_s103
                                 + 1.17603 w_k1_MSFT_s103
                                 + 1.17423 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.869351 w_k1_ADM_s103
                                 + 1.19257 w_k1_AXP_s103
                                 + 0.764372 w_k1_DIS_s103
                                 + 0.858461 w_k1_FDX_s103
                                 + 1.16806 w_k1_IBM_s103
                                 + 0.973224 w_k1_INTC_s103
                                 + 0.865561 w_k1_KO_s103 + 1.0392 w_k1_MAT_s103
                                 + 0.873541 w_k1_MSFT_s103
                                 + 0.92697 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.892442 w_k1_ADM_s103
                                 + 1.19173 w_k1_AXP_s103
                                 + 0.96864 w_k1_DIS_s103
                                 + 0.877925 w_k1_FDX_s103
                                 + 0.889015 w_k1_IBM_s103
                                 + 0.967883 w_k1_INTC_s103
                                 + 0.794797 w_k1_KO_s103
                                 + 1.07236 w_k1_MAT_s103
                                 + 0.887136 w_k1_MSFT_s103
                                 + 0.903554 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.876416 w_k1_ADM_s103
                                 + 0.862597 w_k1_AXP_s103
                                 + 0.817412 w_k1_DIS_s103
                                 + 0.840531 w_k1_FDX_s103
                                 + 0.783449 w_k1_IBM_s103
                                 + 1.02119 w_k1_INTC_s103
                                 + 0.768532 w_k1_KO_s103
                                 + 0.745626 w_k1_MAT_s103
                                 + 0.974872 w_k1_MSFT_s103
                                 + 0.910349 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.847889 w_k1_ADM_s103
                                 + 1.4929 w_k1_AXP_s103 + 1.04141 w_k1_DIS_s103
                                 + 1.02465 w_k1_FDX_s103
                                 + 1.30149 w_k1_IBM_s103
                                 + 0.984731 w_k1_INTC_s103
                                 + 0.961426 w_k1_KO_s103
                                 + 1.58198 w_k1_MAT_s103
                                 + 0.960868 w_k1_MSFT_s103
                                 + 0.9789 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.89725 w_k1_ADM_s103
                                 + 0.773812 w_k1_AXP_s103
                                 + 0.53378 w_k1_DIS_s103
                                 + 0.612182 w_k1_FDX_s103
                                 + 0.666289 w_k1_IBM_s103
                                 + 0.671139 w_k1_INTC_s103
                                 + 0.597087 w_k1_KO_s103
                                 + 0.754858 w_k1_MAT_s103
                                 + 0.61365 w_k1_MSFT_s103
                                 + 0.751499 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.87363 w_k1_ADM_s103
                                 + 1.05813 w_k1_AXP_s103
                                 + 0.906269 w_k1_DIS_s103
                                 + 0.909707 w_k1_FDX_s103
                                 + 0.662442 w_k1_IBM_s103
                                 + 0.837266 w_k1_INTC_s103
                                 + 0.735302 w_k1_KO_s103
                                 + 0.917785 w_k1_MAT_s103
                                 + 0.933197 w_k1_MSFT_s103
                                 + 0.886146 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.23754 w_k1_ADM_s104
                                 + 1.21294 w_k1_AXP_s104
                                 + 0.808844 w_k1_DIS_s104
                                 + 0.981108 w_k1_FDX_s104
                                 + 1.0063 w_k1_IBM_s104
                                 + 1.12089 w_k1_INTC_s104
                                 + 0.993256 w_k1_KO_s104
                                 + 0.996522 w_k1_MAT_s104
                                 + 1.00392 w_k1_MSFT_s104
                                 + 1.05424 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.25006 w_k1_ADM_s104
                                 + 1.47082 w_k1_AXP_s104
                                 + 1.44809 w_k1_DIS_s104
                                 + 1.28192 w_k1_FDX_s104
                                 + 1.37572 w_k1_IBM_s104
                                 + 1.23885 w_k1_INTC_s104
                                 + 1.64664 w_k1_KO_s104 + 1.33618 w_k1_MAT_s104
                                 + 1.19489 w_k1_MSFT_s104
                                 + 1.18741 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.812497 w_k1_ADM_s104
                                 + 0.729401 w_k1_AXP_s104
                                 + 0.639581 w_k1_DIS_s104
                                 + 0.755959 w_k1_FDX_s104
                                 + 0.739226 w_k1_IBM_s104
                                 + 0.987398 w_k1_INTC_s104
                                 + 0.754869 w_k1_KO_s104
                                 + 0.527879 w_k1_MAT_s104
                                 + 0.772012 w_k1_MSFT_s104
                                 + 0.810816 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.925294 w_k1_ADM_s104
                                 + 1.08812 w_k1_AXP_s104
                                 + 0.972438 w_k1_DIS_s104
                                 + 0.909385 w_k1_FDX_s104
                                 + 1.26344 w_k1_IBM_s104
                                 + 0.941388 w_k1_INTC_s104
                                 + 1.17762 w_k1_KO_s104 + 1.3164 w_k1_MAT_s104
                                 + 0.92321 w_k1_MSFT_s104
                                 + 0.996067 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.40509 w_k1_ADM_s104
                                 + 1.3251 w_k1_AXP_s104 + 1.51289 w_k1_DIS_s104
                                 + 1.39672 w_k1_FDX_s104
                                 + 0.913589 w_k1_IBM_s104
                                 + 1.18167 w_k1_INTC_s104
                                 + 1.23984 w_k1_KO_s104
                                 + 0.782372 w_k1_MAT_s104
                                 + 1.20499 w_k1_MSFT_s104 + 1.2469 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.00857 w_k1_ADM_s104
                                 + 0.717037 w_k1_AXP_s104
                                 + 0.88493 w_k1_DIS_s104 + 0.8795 w_k1_FDX_s104
                                 + 0.881396 w_k1_IBM_s104
                                 + 0.871891 w_k1_INTC_s104
                                 + 0.935264 w_k1_KO_s104
                                 + 0.625111 w_k1_MAT_s104
                                 + 0.931132 w_k1_MSFT_s104
                                 + 0.939462 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.00308 w_k1_ADM_s104
                                 + 1.06579 w_k1_AXP_s104
                                 + 1.01825 w_k1_DIS_s104
                                 + 1.09755 w_k1_FDX_s104
                                 + 0.794772 w_k1_IBM_s104
                                 + 1.00894 w_k1_INTC_s104
                                 + 0.978494 w_k1_KO_s104
                                 + 0.752523 w_k1_MAT_s104
                                 + 0.99873 w_k1_MSFT_s104
                                 + 1.04022 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.12307 w_k1_ADM_s104
                                 + 0.911753 w_k1_AXP_s104
                                 + 1.21528 w_k1_DIS_s104
                                 + 1.05824 w_k1_FDX_s104
                                 + 1.21911 w_k1_IBM_s104
                                 + 1.25556 w_k1_INTC_s104
                                 + 1.34449 w_k1_KO_s104
                                 + 0.646929 w_k1_MAT_s104
                                 + 1.10903 w_k1_MSFT_s104
                                 + 1.15333 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.40192 w_k1_ADM_s105
                                 + 1.29083 w_k1_AXP_s105
                                 + 1.53816 w_k1_DIS_s105
                                 + 1.48918 w_k1_FDX_s105
                                 + 1.07858 w_k1_IBM_s105
                                 + 1.31723 w_k1_INTC_s105 + 1.3717 w_k1_KO_s105
                                 + 1.01724 w_k1_MAT_s105
                                 + 1.42546 w_k1_MSFT_s105
                                 + 1.46863 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.33833 w_k1_ADM_s105
                                 + 1.01925 w_k1_AXP_s105
                                 + 0.737967 w_k1_DIS_s105
                                 + 1.12374 w_k1_FDX_s105
                                 + 0.630656 w_k1_IBM_s105
                                 + 1.12063 w_k1_INTC_s105
                                 + 0.712579 w_k1_KO_s105
                                 + 0.611618 w_k1_MAT_s105
                                 + 0.840685 w_k1_MSFT_s105
                                 + 0.890245 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.23472 w_k1_ADM_s105
                                 + 1.20494 w_k1_AXP_s105
                                 + 1.09925 w_k1_DIS_s105
                                 + 1.11911 w_k1_FDX_s105
                                 + 1.37114 w_k1_IBM_s105
                                 + 1.28719 w_k1_INTC_s105
                                 + 1.24134 w_k1_KO_s105 + 1.39928 w_k1_MAT_s105
                                 + 0.999595 w_k1_MSFT_s105
                                 + 1.11108 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.963829 w_k1_ADM_s105
                                 + 1.11999 w_k1_AXP_s105
                                 + 0.659476 w_k1_DIS_s105
                                 + 0.883677 w_k1_FDX_s105
                                 + 1.06005 w_k1_IBM_s105
                                 + 1.20227 w_k1_INTC_s105 + 0.8902 w_k1_KO_s105
                                 + 1.48519 w_k1_MAT_s105
                                 + 0.920956 w_k1_MSFT_s105
                                 + 0.996975 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.1005 w_k1_ADM_s105
                                 + 1.32496 w_k1_AXP_s105 + 1.3666 w_k1_DIS_s105
                                 + 1.07255 w_k1_FDX_s105 + 1.8164 w_k1_IBM_s105
                                 + 1.29113 w_k1_INTC_s105
                                 + 1.31838 w_k1_KO_s105 + 2.23643 w_k1_MAT_s105
                                 + 1.08497 w_k1_MSFT_s105
                                 + 1.13362 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.02949 w_k1_ADM_s105
                                 + 0.74225 w_k1_AXP_s105
                                 + 0.881765 w_k1_DIS_s105
                                 + 1.00731 w_k1_FDX_s105
                                 + 0.701172 w_k1_IBM_s105
                                 + 1.09152 w_k1_INTC_s105
                                 + 0.898882 w_k1_KO_s105
                                 + 0.648571 w_k1_MAT_s105
                                 + 1.07036 w_k1_MSFT_s105
                                 + 1.08767 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.963349 w_k1_ADM_s105
                                 + 1.18016 w_k1_AXP_s105
                                 + 0.867986 w_k1_DIS_s105
                                 + 1.00519 w_k1_FDX_s105
                                 + 0.901718 w_k1_IBM_s105
                                 + 1.29007 w_k1_INTC_s105
                                 + 0.997343 w_k1_KO_s105
                                 + 1.10423 w_k1_MAT_s105 + 0.998 w_k1_MSFT_s105
                                 + 1.12408 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.952066 w_k1_ADM_s105
                                 + 0.773504 w_k1_AXP_s105
                                 + 0.745994 w_k1_DIS_s105
                                 + 0.799425 w_k1_FDX_s105
                                 + 1.04254 w_k1_IBM_s105
                                 + 0.993403 w_k1_INTC_s105
                                 + 0.996698 w_k1_KO_s105
                                 + 0.986751 w_k1_MAT_s105
                                 + 0.893073 w_k1_MSFT_s105
                                 + 1.00854 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.19309 w_k1_ADM_s106
                                 + 0.666112 w_k1_AXP_s106
                                 + 1.07134 w_k1_DIS_s106
                                 + 0.930989 w_k1_FDX_s106
                                 + 0.865212 w_k1_IBM_s106
                                 + 0.81335 w_k1_INTC_s106
                                 + 0.962126 w_k1_KO_s106
                                 + 0.68431 w_k1_MAT_s106
                                 + 0.98036 w_k1_MSFT_s106
                                 + 0.975931 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.988948 w_k1_ADM_s106
                                 + 0.994896 w_k1_AXP_s106
                                 + 0.727339 w_k1_DIS_s106
                                 + 0.908086 w_k1_FDX_s106
                                 + 0.849499 w_k1_IBM_s106
                                 + 0.757509 w_k1_INTC_s106
                                 + 0.895755 w_k1_KO_s106
                                 + 0.793101 w_k1_MAT_s106
                                 + 0.770849 w_k1_MSFT_s106
                                 + 0.910882 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.16949 w_k1_ADM_s106
                                 + 1.43295 w_k1_AXP_s106
                                 + 1.17709 w_k1_DIS_s106
                                 + 1.14636 w_k1_FDX_s106 + 1.5212 w_k1_IBM_s106
                                 + 0.913839 w_k1_INTC_s106
                                 + 1.10758 w_k1_KO_s106 + 1.40792 w_k1_MAT_s106
                                 + 0.948408 w_k1_MSFT_s106
                                 + 1.06301 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.49256 w_k1_ADM_s106
                                 + 1.01327 w_k1_AXP_s106
                                 + 1.67396 w_k1_DIS_s106
                                 + 1.17815 w_k1_FDX_s106 + 1.4971 w_k1_IBM_s106
                                 + 0.827952 w_k1_INTC_s106
                                 + 1.33229 w_k1_KO_s106 + 1.05246 w_k1_MAT_s106
                                 + 1.03115 w_k1_MSFT_s106
                                 + 1.19944 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.32299 w_k1_ADM_s106
                                 + 1.38805 w_k1_AXP_s106
                                 + 1.45407 w_k1_DIS_s106
                                 + 1.24836 w_k1_FDX_s106
                                 + 0.967608 w_k1_IBM_s106
                                 + 1.07312 w_k1_INTC_s106
                                 + 1.12876 w_k1_KO_s106
                                 + 0.871697 w_k1_MAT_s106
                                 + 1.08529 w_k1_MSFT_s106
                                 + 1.13186 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.972003 w_k1_ADM_s106
                                 + 0.768802 w_k1_AXP_s106
                                 + 0.897949 w_k1_DIS_s106
                                 + 0.848424 w_k1_FDX_s106
                                 + 1.24787 w_k1_IBM_s106
                                 + 0.804302 w_k1_INTC_s106
                                 + 1.19343 w_k1_KO_s106
                                 + 0.917427 w_k1_MAT_s106
                                 + 0.99847 w_k1_MSFT_s106
                                 + 1.01196 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.914157 w_k1_ADM_s106
                                 + 0.512771 w_k1_AXP_s106
                                 + 0.688297 w_k1_DIS_s106
                                 + 0.770802 w_k1_FDX_s106
                                 + 0.699274 w_k1_IBM_s106
                                 + 0.790568 w_k1_INTC_s106
                                 + 0.873584 w_k1_KO_s106
                                 + 0.367141 w_k1_MAT_s106
                                 + 0.845822 w_k1_MSFT_s106
                                 + 0.902836 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.31195 w_k1_ADM_s106
                                 + 1.07199 w_k1_AXP_s106 + 1.1569 w_k1_DIS_s106
                                 + 1.09981 w_k1_FDX_s106
                                 + 0.939629 w_k1_IBM_s106
                                 + 0.829539 w_k1_INTC_s106
                                 + 1.10482 w_k1_KO_s106
                                 + 0.675957 w_k1_MAT_s106
                                 + 0.905704 w_k1_MSFT_s106
                                 + 1.13603 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.866566 w_k1_ADM_s107
                                 + 1.03734 w_k1_AXP_s107
                                 + 0.823713 w_k1_DIS_s107
                                 + 0.787374 w_k1_FDX_s107
                                 + 1.41416 w_k1_IBM_s107
                                 + 0.912546 w_k1_INTC_s107
                                 + 0.996526 w_k1_KO_s107
                                 + 1.71737 w_k1_MAT_s107
                                 + 0.796789 w_k1_MSFT_s107
                                 + 0.927916 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.07762 w_k1_ADM_s107
                                 + 1.27317 w_k1_AXP_s107
                                 + 1.43771 w_k1_DIS_s107
                                 + 1.18315 w_k1_FDX_s107
                                 + 1.55497 w_k1_IBM_s107
                                 + 1.23096 w_k1_INTC_s107
                                 + 1.27501 w_k1_KO_s107 + 2.09138 w_k1_MAT_s107
                                 + 1.19748 w_k1_MSFT_s107
                                 + 1.18128 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.45219 w_k1_ADM_s107
                                 + 1.51267 w_k1_AXP_s107 + 1.3378 w_k1_DIS_s107
                                 + 1.31501 w_k1_FDX_s107
                                 + 1.09471 w_k1_IBM_s107
                                 + 1.14901 w_k1_INTC_s107
                                 + 1.19103 w_k1_KO_s107 + 1.40594 w_k1_MAT_s107
                                 + 1.33669 w_k1_MSFT_s107
                                 + 1.16147 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.953804 w_k1_ADM_s107
                                 + 0.881842 w_k1_AXP_s107
                                 + 0.897016 w_k1_DIS_s107
                                 + 0.943824 w_k1_FDX_s107
                                 + 0.771488 w_k1_IBM_s107
                                 + 0.904645 w_k1_INTC_s107
                                 + 0.889238 w_k1_KO_s107
                                 + 1.09152 w_k1_MAT_s107 + 1.012 w_k1_MSFT_s107
                                 + 0.937349 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.842851 w_k1_ADM_s107
                                 + 0.81065 w_k1_AXP_s107
                                 + 0.636304 w_k1_DIS_s107
                                 + 0.713474 w_k1_FDX_s107
                                 + 0.913835 w_k1_IBM_s107
                                 + 0.789188 w_k1_INTC_s107
                                 + 0.721505 w_k1_KO_s107
                                 + 0.914979 w_k1_MAT_s107
                                 + 0.752838 w_k1_MSFT_s107
                                 + 0.7389 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.04027 w_k1_ADM_s107
                                 + 0.972997 w_k1_AXP_s107
                                 + 1.43266 w_k1_DIS_s107 + 1.0921 w_k1_FDX_s107
                                 + 1.15866 w_k1_IBM_s107
                                 + 1.14356 w_k1_INTC_s107 + 1.2392 w_k1_KO_s107
                                 + 1.31975 w_k1_MAT_s107
                                 + 1.17946 w_k1_MSFT_s107
                                 + 1.02561 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.55143 w_k1_ADM_s107
                                 + 1.57868 w_k1_AXP_s107
                                 + 1.67862 w_k1_DIS_s107
                                 + 1.46462 w_k1_FDX_s107
                                 + 1.02155 w_k1_IBM_s107
                                 + 1.16952 w_k1_INTC_s107
                                 + 1.24602 w_k1_KO_s107 + 1.74442 w_k1_MAT_s107
                                 + 1.40644 w_k1_MSFT_s107
                                 + 1.12882 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.20689 w_k1_ADM_s107
                                 + 1.09175 w_k1_AXP_s107
                                 + 1.11529 w_k1_DIS_s107
                                 + 1.11267 w_k1_FDX_s107
                                 + 1.06737 w_k1_IBM_s107
                                 + 1.10625 w_k1_INTC_s107
                                 + 0.984841 w_k1_KO_s107
                                 + 1.20914 w_k1_MAT_s107
                                 + 1.05666 w_k1_MSFT_s107
                                 + 1.06232 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.20891 w_k1_ADM_s110
                                 + 0.918502 w_k1_AXP_s110
                                 + 1.0166 w_k1_DIS_s110
                                 + 0.988916 w_k1_FDX_s110
                                 + 1.41609 w_k1_IBM_s110
                                 + 1.16319 w_k1_INTC_s110
                                 + 1.20403 w_k1_KO_s110
                                 + 0.967625 w_k1_MAT_s110
                                 + 1.01469 w_k1_MSFT_s110
                                 + 1.02269 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.917716 w_k1_ADM_s110
                                 + 0.916254 w_k1_AXP_s110
                                 + 0.775442 w_k1_DIS_s110
                                 + 0.942441 w_k1_FDX_s110
                                 + 0.913882 w_k1_IBM_s110
                                 + 1.12951 w_k1_INTC_s110
                                 + 0.913746 w_k1_KO_s110
                                 + 0.863566 w_k1_MAT_s110
                                 + 0.97124 w_k1_MSFT_s110
                                 + 1.01603 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.96447 w_k1_ADM_s110
                                 + 1.15189 w_k1_AXP_s110
                                 + 0.651276 w_k1_DIS_s110
                                 + 0.849345 w_k1_FDX_s110
                                 + 1.24524 w_k1_IBM_s110
                                 + 1.03972 w_k1_INTC_s110
                                 + 0.831297 w_k1_KO_s110
                                 + 1.32712 w_k1_MAT_s110
                                 + 0.803321 w_k1_MSFT_s110
                                 + 0.840101 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.892082 w_k1_ADM_s110
                                 + 0.784118 w_k1_AXP_s110
                                 + 0.63461 w_k1_DIS_s110
                                 + 0.94087 w_k1_FDX_s110
                                 + 0.617235 w_k1_IBM_s110
                                 + 1.17321 w_k1_INTC_s110
                                 + 0.668074 w_k1_KO_s110
                                 + 0.536924 w_k1_MAT_s110
                                 + 0.966963 w_k1_MSFT_s110
                                 + 0.853635 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.951754 w_k1_ADM_s110
                                 + 0.891847 w_k1_AXP_s110
                                 + 1.04171 w_k1_DIS_s110
                                 + 1.14047 w_k1_FDX_s110
                                 + 0.713932 w_k1_IBM_s110
                                 + 1.14286 w_k1_INTC_s110
                                 + 0.906294 w_k1_KO_s110
                                 + 0.87289 w_k1_MAT_s110
                                 + 1.21586 w_k1_MSFT_s110
                                 + 1.01515 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.53751 w_k1_ADM_s110
                                 + 1.39252 w_k1_AXP_s110
                                 + 1.45922 w_k1_DIS_s110
                                 + 1.34876 w_k1_FDX_s110
                                 + 1.09104 w_k1_IBM_s110
                                 + 1.35082 w_k1_INTC_s110 + 1.1907 w_k1_KO_s110
                                 + 1.22007 w_k1_MAT_s110
                                 + 1.21212 w_k1_MSFT_s110
                                 + 1.21434 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.2728 w_k1_ADM_s110
                                 + 1.56348 w_k1_AXP_s110
                                 + 1.12722 w_k1_DIS_s110
                                 + 1.34716 w_k1_FDX_s110
                                 + 0.785608 w_k1_IBM_s110
                                 + 1.41197 w_k1_INTC_s110
                                 + 0.885896 w_k1_KO_s110
                                 + 0.898439 w_k1_MAT_s110
                                 + 1.31482 w_k1_MSFT_s110
                                 + 0.960458 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.829106 w_k1_ADM_s110
                                 + 0.748234 w_k1_AXP_s110
                                 + 0.813747 w_k1_DIS_s110
                                 + 0.980396 w_k1_FDX_s110
                                 + 0.717636 w_k1_IBM_s110
                                 + 1.02335 w_k1_INTC_s110
                                 + 0.622322 w_k1_KO_s110
                                 + 0.69047 w_k1_MAT_s110
                                 + 0.911467 w_k1_MSFT_s110
                                 + 0.840074 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.597817 w_k1_ADM_s111
                                 + 0.531064 w_k1_AXP_s111
                                 + 0.537274 w_k1_DIS_s111
                                 + 0.682128 w_k1_FDX_s111
                                 + 0.672998 w_k1_IBM_s111
                                 + 0.770296 w_k1_INTC_s111
                                 + 0.632233 w_k1_KO_s111
                                 + 0.44728 w_k1_MAT_s111
                                 + 0.685525 w_k1_MSFT_s111
                                 + 0.73374 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.776074 w_k1_ADM_s111
                                 + 0.915939 w_k1_AXP_s111
                                 + 0.562868 w_k1_DIS_s111
                                 + 0.795928 w_k1_FDX_s111
                                 + 0.882477 w_k1_IBM_s111
                                 + 0.867546 w_k1_INTC_s111
                                 + 0.685944 w_k1_KO_s111
                                 + 0.888795 w_k1_MAT_s111
                                 + 0.740607 w_k1_MSFT_s111
                                 + 0.775616 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.979664 w_k1_ADM_s111
                                 + 1.03344 w_k1_AXP_s111
                                 + 0.983505 w_k1_DIS_s111
                                 + 1.02251 w_k1_FDX_s111
                                 + 0.932559 w_k1_IBM_s111
                                 + 0.84313 w_k1_INTC_s111
                                 + 0.917818 w_k1_KO_s111
                                 + 0.943412 w_k1_MAT_s111
                                 + 0.976303 w_k1_MSFT_s111
                                 + 0.99335 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.861884 w_k1_ADM_s111
                                 + 1.43811 w_k1_AXP_s111
                                 + 1.64669 w_k1_DIS_s111
                                 + 1.37085 w_k1_FDX_s111
                                 + 1.31781 w_k1_IBM_s111
                                 + 1.33174 w_k1_INTC_s111
                                 + 1.15445 w_k1_KO_s111 + 1.33744 w_k1_MAT_s111
                                 + 1.05298 w_k1_MSFT_s111
                                 + 1.22879 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.850836 w_k1_ADM_s111
                                 + 0.831457 w_k1_AXP_s111
                                 + 0.892128 w_k1_DIS_s111
                                 + 0.932427 w_k1_FDX_s111
                                 + 0.766827 w_k1_IBM_s111
                                 + 0.872983 w_k1_INTC_s111
                                 + 0.796585 w_k1_KO_s111
                                 + 0.510458 w_k1_MAT_s111
                                 + 0.987964 w_k1_MSFT_s111
                                 + 0.89134 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.750798 w_k1_ADM_s111
                                 + 1.01729 w_k1_AXP_s111
                                 + 0.939942 w_k1_DIS_s111
                                 + 0.916571 w_k1_FDX_s111
                                 + 1.3248 w_k1_IBM_s111
                                 + 0.927061 w_k1_INTC_s111
                                 + 0.841805 w_k1_KO_s111
                                 + 1.20623 w_k1_MAT_s111
                                 + 0.895899 w_k1_MSFT_s111
                                 + 0.97336 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.82901 w_k1_ADM_s111
                                 + 0.773697 w_k1_AXP_s111
                                 + 0.691461 w_k1_DIS_s111
                                 + 0.823739 w_k1_FDX_s111 + 0.67 w_k1_IBM_s111
                                 + 0.769244 w_k1_INTC_s111
                                 + 0.722552 w_k1_KO_s111
                                 + 0.463822 w_k1_MAT_s111
                                 + 0.816836 w_k1_MSFT_s111
                                 + 0.844709 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.759035 w_k1_ADM_s111
                                 + 0.789225 w_k1_AXP_s111
                                 + 0.873677 w_k1_DIS_s111
                                 + 0.771417 w_k1_FDX_s111
                                 + 1.0038 w_k1_IBM_s111
                                 + 0.800662 w_k1_INTC_s111
                                 + 0.883876 w_k1_KO_s111
                                 + 0.918934 w_k1_MAT_s111
                                 + 0.896463 w_k1_MSFT_s111
                                 + 0.874392 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.17736 w_k1_ADM_s112
                                 + 0.960757 w_k1_AXP_s112
                                 + 0.894154 w_k1_DIS_s112
                                 + 1.00704 w_k1_FDX_s112
                                 + 0.97244 w_k1_IBM_s112
                                 + 1.1635 w_k1_INTC_s112
                                 + 0.984904 w_k1_KO_s112
                                 + 0.780854 w_k1_MAT_s112
                                 + 0.977754 w_k1_MSFT_s112
                                 + 1.0167 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.28943 w_k1_ADM_s112
                                 + 0.867251 w_k1_AXP_s112
                                 + 0.976343 w_k1_DIS_s112
                                 + 1.09209 w_k1_FDX_s112
                                 + 0.687367 w_k1_IBM_s112
                                 + 1.24577 w_k1_INTC_s112
                                 + 1.01882 w_k1_KO_s112
                                 + 0.537278 w_k1_MAT_s112
                                 + 1.18484 w_k1_MSFT_s112
                                 + 1.10221 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.18713 w_k1_ADM_s112
                                 + 1.17949 w_k1_AXP_s112 + 1.284 w_k1_DIS_s112
                                 + 1.09819 w_k1_FDX_s112
                                 + 1.50077 w_k1_IBM_s112
                                 + 1.30144 w_k1_INTC_s112
                                 + 1.51261 w_k1_KO_s112 + 1.49368 w_k1_MAT_s112
                                 + 1.14236 w_k1_MSFT_s112
                                 + 1.34918 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.01632 w_k1_ADM_s112
                                 + 1.12441 w_k1_AXP_s112
                                 + 1.23121 w_k1_DIS_s112
                                 + 1.10904 w_k1_FDX_s112
                                 + 0.998181 w_k1_IBM_s112
                                 + 1.23649 w_k1_INTC_s112
                                 + 1.17579 w_k1_KO_s112 + 1.45301 w_k1_MAT_s112
                                 + 1.17171 w_k1_MSFT_s112
                                 + 1.15041 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.34399 w_k1_ADM_s112
                                 + 1.36284 w_k1_AXP_s112
                                 + 1.26676 w_k1_DIS_s112
                                 + 1.34897 w_k1_FDX_s112
                                 + 1.00331 w_k1_IBM_s112
                                 + 1.18735 w_k1_INTC_s112
                                 + 1.03484 w_k1_KO_s112 + 1.14889 w_k1_MAT_s112
                                 + 1.16336 w_k1_MSFT_s112
                                 + 1.12883 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.26708 w_k1_ADM_s112
                                 + 1.56105 w_k1_AXP_s112
                                 + 1.81733 w_k1_DIS_s112
                                 + 1.60012 w_k1_FDX_s112 + 1.3853 w_k1_IBM_s112
                                 + 1.30033 w_k1_INTC_s112 + 1.4693 w_k1_KO_s112
                                 + 2.1708 w_k1_MAT_s112
                                 + 1.25678 w_k1_MSFT_s112
                                 + 1.43229 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.07727 w_k1_ADM_s112
                                 + 0.75518 w_k1_AXP_s112
                                 + 0.737603 w_k1_DIS_s112
                                 + 0.762418 w_k1_FDX_s112
                                 + 0.9828 w_k1_IBM_s112
                                 + 0.967005 w_k1_INTC_s112
                                 + 1.00014 w_k1_KO_s112
                                 + 0.779397 w_k1_MAT_s112
                                 + 0.830161 w_k1_MSFT_s112
                                 + 0.84481 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.982615 w_k1_ADM_s112
                                 + 0.967798 w_k1_AXP_s112
                                 + 0.83095 w_k1_DIS_s112
                                 + 0.908325 w_k1_FDX_s112
                                 + 0.90443 w_k1_IBM_s112
                                 + 0.957865 w_k1_INTC_s112
                                 + 0.990719 w_k1_KO_s112
                                 + 0.970308 w_k1_MAT_s112
                                 + 0.961973 w_k1_MSFT_s112
                                 + 0.950802 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.841443 w_k1_ADM_s113
                                 + 0.898053 w_k1_AXP_s113
                                 + 0.753624 w_k1_DIS_s113
                                 + 0.819722 w_k1_FDX_s113
                                 + 0.997007 w_k1_IBM_s113
                                 + 0.726159 w_k1_INTC_s113
                                 + 0.840573 w_k1_KO_s113
                                 + 1.15563 w_k1_MAT_s113
                                 + 0.882296 w_k1_MSFT_s113
                                 + 1.04495 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.947526 w_k1_ADM_s113
                                 + 1.06849 w_k1_AXP_s113
                                 + 0.96863 w_k1_DIS_s113
                                 + 0.823433 w_k1_FDX_s113
                                 + 1.28382 w_k1_IBM_s113
                                 + 1.00415 w_k1_INTC_s113
                                 + 1.05125 w_k1_KO_s113 + 1.18539 w_k1_MAT_s113
                                 + 0.901787 w_k1_MSFT_s113
                                 + 0.976551 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.16404 w_k1_ADM_s113
                                 + 1.15122 w_k1_AXP_s113
                                 + 1.48564 w_k1_DIS_s113
                                 + 1.11947 w_k1_FDX_s113
                                 + 1.90428 w_k1_IBM_s113
                                 + 1.01083 w_k1_INTC_s113
                                 + 1.35327 w_k1_KO_s113 + 1.73495 w_k1_MAT_s113
                                 + 1.16879 w_k1_MSFT_s113
                                 + 1.13892 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.999002 w_k1_ADM_s113
                                 + 0.905935 w_k1_AXP_s113
                                 + 0.832635 w_k1_DIS_s113
                                 + 0.855333 w_k1_FDX_s113
                                 + 0.824552 w_k1_IBM_s113
                                 + 0.726659 w_k1_INTC_s113
                                 + 0.813244 w_k1_KO_s113
                                 + 0.821738 w_k1_MAT_s113
                                 + 0.873153 w_k1_MSFT_s113
                                 + 0.952525 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.1325 w_k1_ADM_s113
                                 + 0.96555 w_k1_AXP_s113 + 1.163 w_k1_DIS_s113
                                 + 1.07149 w_k1_FDX_s113
                                 + 0.990493 w_k1_IBM_s113
                                 + 0.98691 w_k1_INTC_s113
                                 + 1.01052 w_k1_KO_s113 + 1.30826 w_k1_MAT_s113
                                 + 1.20073 w_k1_MSFT_s113
                                 + 1.18912 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.03351 w_k1_ADM_s113
                                 + 1.40076 w_k1_AXP_s113
                                 + 1.52675 w_k1_DIS_s113
                                 + 1.14814 w_k1_FDX_s113
                                 + 1.29851 w_k1_IBM_s113
                                 + 1.10537 w_k1_INTC_s113 + 1.2516 w_k1_KO_s113
                                 + 1.3358 w_k1_MAT_s113
                                 + 1.17468 w_k1_MSFT_s113
                                 + 1.25571 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.681243 w_k1_ADM_s113
                                 + 0.750634 w_k1_AXP_s113
                                 + 0.759002 w_k1_DIS_s113
                                 + 0.899211 w_k1_FDX_s113
                                 + 0.620965 w_k1_IBM_s113
                                 + 0.88766 w_k1_INTC_s113
                                 + 0.782844 w_k1_KO_s113
                                 + 0.652731 w_k1_MAT_s113
                                 + 0.870165 w_k1_MSFT_s113
                                 + 0.890289 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.3585 w_k1_ADM_s113
                                 + 1.35019 w_k1_AXP_s113
                                 + 1.28996 w_k1_DIS_s113
                                 + 1.26517 w_k1_FDX_s113
                                 + 0.899686 w_k1_IBM_s113
                                 + 0.973782 w_k1_INTC_s113
                                 + 1.08793 w_k1_KO_s113 + 1.37394 w_k1_MAT_s113
                                 + 1.14854 w_k1_MSFT_s113 + 1.2142 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.876545 w_k1_ADM_s114
                                 + 0.916018 w_k1_AXP_s114
                                 + 0.981804 w_k1_DIS_s114
                                 + 1.0959 w_k1_FDX_s114
                                 + 0.771238 w_k1_IBM_s114
                                 + 1.09608 w_k1_INTC_s114
                                 + 0.943711 w_k1_KO_s114
                                 + 1.00807 w_k1_MAT_s114
                                 + 1.1116 w_k1_MSFT_s114 + 1.06661 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.10135 w_k1_ADM_s114
                                 + 1.43095 w_k1_AXP_s114 + 1.6834 w_k1_DIS_s114
                                 + 1.33933 w_k1_FDX_s114 + 1.4042 w_k1_IBM_s114
                                 + 1.11309 w_k1_INTC_s114 + 1.4871 w_k1_KO_s114
                                 + 1.83674 w_k1_MAT_s114
                                 + 1.3126 w_k1_MSFT_s114 + 1.25148 w_k1_PG_s114
                                 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.10967 w_k1_ADM_s114
                                 + 1.39686 w_k1_AXP_s114
                                 + 1.33985 w_k1_DIS_s114
                                 + 1.22637 w_k1_FDX_s114
                                 + 1.16223 w_k1_IBM_s114
                                 + 0.944933 w_k1_INTC_s114
                                 + 1.22465 w_k1_KO_s114 + 1.21497 w_k1_MAT_s114
                                 + 1.07158 w_k1_MSFT_s114
                                 + 1.08147 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.899121 w_k1_ADM_s114
                                 + 0.827115 w_k1_AXP_s114
                                 + 1.07622 w_k1_DIS_s114
                                 + 0.984317 w_k1_FDX_s114
                                 + 1.0355 w_k1_IBM_s114
                                 + 0.990992 w_k1_INTC_s114
                                 + 1.13349 w_k1_KO_s114
                                 + 0.872622 w_k1_MAT_s114
                                 + 1.05418 w_k1_MSFT_s114 + 1.0443 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.869284 w_k1_ADM_s114
                                 + 0.978441 w_k1_AXP_s114
                                 + 0.876662 w_k1_DIS_s114
                                 + 0.828654 w_k1_FDX_s114
                                 + 1.39536 w_k1_IBM_s114
                                 + 0.866177 w_k1_INTC_s114
                                 + 1.1267 w_k1_KO_s114 + 1.50705 w_k1_MAT_s114
                                 + 0.951716 w_k1_MSFT_s114
                                 + 1.02092 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.918472 w_k1_ADM_s114
                                 + 1.13066 w_k1_AXP_s114
                                 + 1.19603 w_k1_DIS_s114
                                 + 1.01019 w_k1_FDX_s114
                                 + 1.34249 w_k1_IBM_s114
                                 + 1.15814 w_k1_INTC_s114
                                 + 1.30926 w_k1_KO_s114 + 1.17525 w_k1_MAT_s114
                                 + 1.10924 w_k1_MSFT_s114
                                 + 1.14688 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.29145 w_k1_ADM_s114
                                 + 1.15663 w_k1_AXP_s114
                                 + 1.26791 w_k1_DIS_s114
                                 + 1.21398 w_k1_FDX_s114
                                 + 0.865741 w_k1_IBM_s114
                                 + 1.30259 w_k1_INTC_s114
                                 + 1.04589 w_k1_KO_s114
                                 + 0.660253 w_k1_MAT_s114
                                 + 1.09889 w_k1_MSFT_s114
                                 + 1.03735 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.7433 w_k1_ADM_s114
                                 + 0.764501 w_k1_AXP_s114
                                 + 0.763014 w_k1_DIS_s114
                                 + 0.802228 w_k1_FDX_s114
                                 + 0.816197 w_k1_IBM_s114
                                 + 0.789461 w_k1_INTC_s114
                                 + 0.779378 w_k1_KO_s114
                                 + 0.779844 w_k1_MAT_s114
                                 + 0.833971 w_k1_MSFT_s114
                                 + 0.849132 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.877967 w_k1_ADM_s115
                                 + 1.01346 w_k1_AXP_s115
                                 + 0.981643 w_k1_DIS_s115
                                 + 0.908663 w_k1_FDX_s115
                                 + 0.99587 w_k1_IBM_s115
                                 + 0.924684 w_k1_INTC_s115
                                 + 0.91674 w_k1_KO_s115 + 1.20328 w_k1_MAT_s115
                                 + 0.777959 w_k1_MSFT_s115
                                 + 0.86053 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.945003 w_k1_ADM_s115
                                 + 0.902275 w_k1_AXP_s115
                                 + 0.801557 w_k1_DIS_s115
                                 + 0.850035 w_k1_FDX_s115
                                 + 1.2728 w_k1_IBM_s115
                                 + 0.930439 w_k1_INTC_s115
                                 + 0.962974 w_k1_KO_s115
                                 + 1.34665 w_k1_MAT_s115
                                 + 0.848635 w_k1_MSFT_s115
                                 + 1.07474 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.873751 w_k1_ADM_s115
                                 + 0.806873 w_k1_AXP_s115
                                 + 0.654131 w_k1_DIS_s115
                                 + 0.811524 w_k1_FDX_s115
                                 + 0.744717 w_k1_IBM_s115
                                 + 0.858933 w_k1_INTC_s115
                                 + 0.718013 w_k1_KO_s115
                                 + 0.735223 w_k1_MAT_s115
                                 + 0.791542 w_k1_MSFT_s115
                                 + 0.815888 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.882021 w_k1_ADM_s115
                                 + 1.04624 w_k1_AXP_s115
                                 + 1.21381 w_k1_DIS_s115
                                 + 1.04736 w_k1_FDX_s115
                                 + 1.17007 w_k1_IBM_s115
                                 + 1.02453 w_k1_INTC_s115
                                 + 1.01062 w_k1_KO_s115 + 1.06225 w_k1_MAT_s115
                                 + 1.03848 w_k1_MSFT_s115
                                 + 1.02757 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.16076 w_k1_ADM_s115
                                 + 1.08666 w_k1_AXP_s115
                                 + 1.20122 w_k1_DIS_s115
                                 + 1.20984 w_k1_FDX_s115
                                 + 0.856863 w_k1_IBM_s115
                                 + 1.03042 w_k1_INTC_s115
                                 + 0.988767 w_k1_KO_s115 + 1.1043 w_k1_MAT_s115
                                 + 0.995026 w_k1_MSFT_s115
                                 + 1.06398 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.999667 w_k1_ADM_s115
                                 + 1.06384 w_k1_AXP_s115
                                 + 0.877758 w_k1_DIS_s115
                                 + 0.992291 w_k1_FDX_s115
                                 + 0.931968 w_k1_IBM_s115
                                 + 0.91572 w_k1_INTC_s115
                                 + 0.969264 w_k1_KO_s115
                                 + 1.42278 w_k1_MAT_s115
                                 + 0.825074 w_k1_MSFT_s115
                                 + 0.95867 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.854435 w_k1_ADM_s115
                                 + 0.690961 w_k1_AXP_s115
                                 + 0.507153 w_k1_DIS_s115
                                 + 0.648464 w_k1_FDX_s115
                                 + 0.927607 w_k1_IBM_s115
                                 + 0.733944 w_k1_INTC_s115
                                 + 0.748805 w_k1_KO_s115
                                 + 1.01019 w_k1_MAT_s115
                                 + 0.713861 w_k1_MSFT_s115
                                 + 0.811533 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.36748 w_k1_ADM_s115
                                 + 1.51586 w_k1_AXP_s115
                                 + 1.66298 w_k1_DIS_s115
                                 + 1.46223 w_k1_FDX_s115
                                 + 1.33453 w_k1_IBM_s115 + 1.436 w_k1_INTC_s115
                                 + 1.39922 w_k1_KO_s115 + 1.89938 w_k1_MAT_s115
                                 + 1.3545 w_k1_MSFT_s115 + 1.205 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.818113 w_k1_ADM_s116
                                 + 0.805355 w_k1_AXP_s116
                                 + 0.912663 w_k1_DIS_s116
                                 + 0.91379 w_k1_FDX_s116
                                 + 0.837625 w_k1_IBM_s116
                                 + 0.966374 w_k1_INTC_s116
                                 + 0.820318 w_k1_KO_s116
                                 + 0.769796 w_k1_MAT_s116
                                 + 1.00224 w_k1_MSFT_s116
                                 + 0.890461 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.17073 w_k1_ADM_s116
                                 + 1.01816 w_k1_AXP_s116
                                 + 1.19443 w_k1_DIS_s116
                                 + 1.04959 w_k1_FDX_s116
                                 + 1.17472 w_k1_IBM_s116
                                 + 1.01864 w_k1_INTC_s116 + 1.1098 w_k1_KO_s116
                                 + 1.0872 w_k1_MAT_s116
                                 + 1.06047 w_k1_MSFT_s116
                                 + 1.11545 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.985794 w_k1_ADM_s116
                                 + 0.809683 w_k1_AXP_s116
                                 + 1.01059 w_k1_DIS_s116
                                 + 0.854562 w_k1_FDX_s116
                                 + 1.17406 w_k1_IBM_s116
                                 + 0.799244 w_k1_INTC_s116
                                 + 1.11139 w_k1_KO_s116 + 1.0389 w_k1_MAT_s116
                                 + 0.955309 w_k1_MSFT_s116
                                 + 0.998045 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.935088 w_k1_ADM_s116
                                 + 0.89634 w_k1_AXP_s116
                                 + 0.865167 w_k1_DIS_s116
                                 + 0.897991 w_k1_FDX_s116
                                 + 0.604912 w_k1_IBM_s116
                                 + 0.888057 w_k1_INTC_s116
                                 + 0.697609 w_k1_KO_s116
                                 + 0.575734 w_k1_MAT_s116
                                 + 0.923982 w_k1_MSFT_s116
                                 + 0.862178 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.18321 w_k1_ADM_s116
                                 + 1.00517 w_k1_AXP_s116
                                 + 1.25072 w_k1_DIS_s116
                                 + 1.05716 w_k1_FDX_s116
                                 + 0.88949 w_k1_IBM_s116
                                 + 1.10278 w_k1_INTC_s116
                                 + 1.01835 w_k1_KO_s116
                                 + 0.669196 w_k1_MAT_s116
                                 + 1.15318 w_k1_MSFT_s116
                                 + 0.963099 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.2145 w_k1_ADM_s116
                                 + 1.35553 w_k1_AXP_s116
                                 + 1.66555 w_k1_DIS_s116
                                 + 1.22184 w_k1_FDX_s116
                                 + 1.55458 w_k1_IBM_s116
                                 + 1.28558 w_k1_INTC_s116
                                 + 1.43824 w_k1_KO_s116 + 1.18925 w_k1_MAT_s116
                                 + 1.23455 w_k1_MSFT_s116
                                 + 1.32796 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.15101 w_k1_ADM_s116
                                 + 1.20672 w_k1_AXP_s116
                                 + 1.22078 w_k1_DIS_s116
                                 + 1.19705 w_k1_FDX_s116
                                 + 1.10742 w_k1_IBM_s116
                                 + 1.24841 w_k1_INTC_s116
                                 + 1.21478 w_k1_KO_s116
                                 + 0.599724 w_k1_MAT_s116
                                 + 1.20766 w_k1_MSFT_s116
                                 + 0.942416 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.39538 w_k1_ADM_s116
                                 + 1.27321 w_k1_AXP_s116
                                 + 1.84141 w_k1_DIS_s116
                                 + 1.27611 w_k1_FDX_s116
                                 + 1.24641 w_k1_IBM_s116
                                 + 1.09757 w_k1_INTC_s116
                                 + 1.50691 w_k1_KO_s116 + 1.07294 w_k1_MAT_s116
                                 + 1.49488 w_k1_MSFT_s116
                                 + 1.18235 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.18324 w_k1_ADM_s117
                                 + 1.34777 w_k1_AXP_s117
                                 + 1.22691 w_k1_DIS_s117
                                 + 1.19322 w_k1_FDX_s117 + 1.1577 w_k1_IBM_s117
                                 + 1.21084 w_k1_INTC_s117
                                 + 1.30931 w_k1_KO_s117 + 1.23269 w_k1_MAT_s117
                                 + 1.15023 w_k1_MSFT_s117 + 1.1487 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.996999 w_k1_ADM_s117
                                 + 1.20118 w_k1_AXP_s117
                                 + 0.715482 w_k1_DIS_s117
                                 + 0.825724 w_k1_FDX_s117
                                 + 1.24358 w_k1_IBM_s117
                                 + 0.839117 w_k1_INTC_s117
                                 + 0.895335 w_k1_KO_s117
                                 + 1.45574 w_k1_MAT_s117
                                 + 0.825067 w_k1_MSFT_s117
                                 + 0.928869 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.26096 w_k1_ADM_s117
                                 + 1.56583 w_k1_AXP_s117
                                 + 1.71441 w_k1_DIS_s117
                                 + 1.41476 w_k1_FDX_s117
                                 + 1.44699 w_k1_IBM_s117
                                 + 1.02575 w_k1_INTC_s117
                                 + 1.57813 w_k1_KO_s117 + 1.80449 w_k1_MAT_s117
                                 + 1.25163 w_k1_MSFT_s117
                                 + 1.33656 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.28033 w_k1_ADM_s117
                                 + 0.955786 w_k1_AXP_s117
                                 + 0.874139 w_k1_DIS_s117
                                 + 1.03676 w_k1_FDX_s117
                                 + 0.747331 w_k1_IBM_s117
                                 + 0.879354 w_k1_INTC_s117
                                 + 0.902668 w_k1_KO_s117
                                 + 0.616052 w_k1_MAT_s117
                                 + 1.02817 w_k1_MSFT_s117
                                 + 0.980106 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.04434 w_k1_ADM_s117
                                 + 1.08392 w_k1_AXP_s117
                                 + 1.23524 w_k1_DIS_s117
                                 + 1.03756 w_k1_FDX_s117
                                 + 1.44526 w_k1_IBM_s117
                                 + 1.02491 w_k1_INTC_s117
                                 + 1.33654 w_k1_KO_s117 + 1.06452 w_k1_MAT_s117
                                 + 1.06422 w_k1_MSFT_s117
                                 + 1.15265 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.23825 w_k1_ADM_s117
                                 + 1.14635 w_k1_AXP_s117
                                 + 1.35245 w_k1_DIS_s117
                                 + 1.33687 w_k1_FDX_s117
                                 + 0.878036 w_k1_IBM_s117
                                 + 1.11525 w_k1_INTC_s117
                                 + 1.11595 w_k1_KO_s117 + 0.90769 w_k1_MAT_s117
                                 + 1.14416 w_k1_MSFT_s117
                                 + 1.12044 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.739253 w_k1_ADM_s117
                                 + 0.636611 w_k1_AXP_s117
                                 + 0.635896 w_k1_DIS_s117
                                 + 0.775438 w_k1_FDX_s117
                                 + 0.741679 w_k1_IBM_s117
                                 + 0.850064 w_k1_INTC_s117
                                 + 0.846706 w_k1_KO_s117
                                 + 0.764874 w_k1_MAT_s117
                                 + 0.869996 w_k1_MSFT_s117
                                 + 0.903657 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.777644 w_k1_ADM_s117
                                 + 0.833348 w_k1_AXP_s117
                                 + 0.99373 w_k1_DIS_s117
                                 + 0.957123 w_k1_FDX_s117
                                 + 0.985122 w_k1_IBM_s117
                                 + 0.876943 w_k1_INTC_s117
                                 + 0.870168 w_k1_KO_s117
                                 + 0.935162 w_k1_MAT_s117
                                 + 0.953207 w_k1_MSFT_s117
                                 + 0.95569 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0987297350174828 w_k1_ADM_s000
                                 + 0.0739449403359638 w_k1_ADM_s001
                                 + 0.0758422747004919 w_k1_ADM_s002
                                 + 0.0226419062630471 w_k1_ADM_s003
                                 + 0.0275756118004428 w_k1_ADM_s004
                                 + 0.028615969105877 w_k1_ADM_s005
                                 + 0.0374170575957341 w_k1_ADM_s006
                                 + 0.0561674090319835 w_k1_ADM_s007
                                 + 0.00928621975075964 w_k1_ADM_s010
                                 + 0.0154843050214344 w_k1_ADM_s011
                                 + 0.0145639316827401 w_k1_ADM_s012
                                 + 0.0145161473572208 w_k1_ADM_s013
                                 + 0.0107416253180714 w_k1_ADM_s014
                                 + 0.0104285647012665 w_k1_ADM_s015
                                 + 0.0270151176298438 w_k1_ADM_s016
                                 + 0.0112878569199063 w_k1_ADM_s017
                                 + 0.0536822372155804 w_k1_ADM_s100
                                 + 0.0363355410232132 w_k1_ADM_s101
                                 + 0.0230557878703202 w_k1_ADM_s102
                                 + 0.0134699545228443 w_k1_ADM_s103
                                 + 0.0278713670275441 w_k1_ADM_s104
                                 + 0.0129085225783145 w_k1_ADM_s105
                                 + 0.0211301800562558 w_k1_ADM_s106
                                 + 0.0428938447248927 w_k1_ADM_s107
                                 + 0.0361767725385718 w_k1_ADM_s110
                                 + 0.0351288920285912 w_k1_ADM_s111
                                 + 0.0176507108904105 w_k1_ADM_s112
                                 + 0.0300908451779222 w_k1_ADM_s113
                                 + 0.0487116801087309 w_k1_ADM_s114
                                 + 0.0398930553734521 w_k1_ADM_s115
                                 + 0.0351767121940714 w_k1_ADM_s116
                                 + 0.0126659033851276 w_k1_ADM_s117
                                 + 0.096725039407522 w_k1_AXP_s000
                                 + 0.0765994479425435 w_k1_AXP_s001
                                 + 0.0688887539163247 w_k1_AXP_s002
                                 + 0.0261252920706968 w_k1_AXP_s003
                                 + 0.0268636211802264 w_k1_AXP_s004
                                 + 0.0303995308925929 w_k1_AXP_s005
                                 + 0.0373719930324171 w_k1_AXP_s006
                                 + 0.0570683124731283 w_k1_AXP_s007
                                 + 0.0090556227226211 w_k1_AXP_s010
                                 + 0.0176896592245307 w_k1_AXP_s011
                                 + 0.0119447987823738 w_k1_AXP_s012
                                 + 0.0131124983153575 w_k1_AXP_s013
                                 + 0.00980430695320449 w_k1_AXP_s014
                                 + 0.00910007277622513 w_k1_AXP_s015
                                 + 0.0274279754956153 w_k1_AXP_s016
                                 + 0.0116449724600842 w_k1_AXP_s017
                                 + 0.0524949839554737 w_k1_AXP_s100
                                 + 0.0339943513349767 w_k1_AXP_s101
                                 + 0.025132294088872 w_k1_AXP_s102
                                 + 0.0167770499604185 w_k1_AXP_s103
                                 + 0.0263849626285814 w_k1_AXP_s104
                                 + 0.012123287260766 w_k1_AXP_s105
                                 + 0.0177528975398291 w_k1_AXP_s106
                                 + 0.042965136048284 w_k1_AXP_s107
                                 + 0.0351308046322292 w_k1_AXP_s110
                                 + 0.0425576931465036 w_k1_AXP_s111
                                 + 0.0161101233235134 w_k1_AXP_s112
                                 + 0.0315394864753611 w_k1_AXP_s113
                                 + 0.0528939777626187 w_k1_AXP_s114
                                 + 0.0404080438843988 w_k1_AXP_s115
                                 + 0.0326254006705478 w_k1_AXP_s116
                                 + 0.0128601502833423 w_k1_AXP_s117
                                 + 0.0903031211120979 w_k1_DIS_s000
                                 + 0.0723387903426973 w_k1_DIS_s001
                                 + 0.0748379737757098 w_k1_DIS_s002
                                 + 0.0270232429575114 w_k1_DIS_s003
                                 + 0.0264716548263206 w_k1_DIS_s004
                                 + 0.0296852960555477 w_k1_DIS_s005
                                 + 0.0363539319340583 w_k1_DIS_s006
                                 + 0.060640748844524 w_k1_DIS_s007
                                 + 0.00922627454298217 w_k1_DIS_s010
                                 + 0.0191870728875455 w_k1_DIS_s011
                                 + 0.0136300176185462 w_k1_DIS_s012
                                 + 0.013920058826685 w_k1_DIS_s013
                                 + 0.00929982362570778 w_k1_DIS_s014
                                 + 0.00972450064949561 w_k1_DIS_s015
                                 + 0.0261909908906912 w_k1_DIS_s016
                                 + 0.0114068094439164 w_k1_DIS_s017
                                 + 0.0528026278546362 w_k1_DIS_s100
                                 + 0.0387178073159743 w_k1_DIS_s101
                                 + 0.0256725984615614 w_k1_DIS_s102
                                 + 0.0148105377540809 w_k1_DIS_s103
                                 + 0.0268097461634804 w_k1_DIS_s104
                                 + 0.0116229449301845 w_k1_DIS_s105
                                 + 0.0196647090208794 w_k1_DIS_s106
                                 + 0.0433335407166058 w_k1_DIS_s107
                                 + 0.0313145217509932 w_k1_DIS_s110
                                 + 0.0428733312705613 w_k1_DIS_s111
                                 + 0.0162856165823265 w_k1_DIS_s112
                                 + 0.0317430015284021 w_k1_DIS_s113
                                 + 0.055962013945438 w_k1_DIS_s114
                                 + 0.0392288026153062 w_k1_DIS_s115
                                 + 0.03893448934586 w_k1_DIS_s116
                                 + 0.0137128549995668 w_k1_DIS_s117
                                 + 0.0930783632475114 w_k1_FDX_s000
                                 + 0.0751226217513276 w_k1_FDX_s001
                                 + 0.0719382389308417 w_k1_FDX_s002
                                 + 0.0258595201931557 w_k1_FDX_s003
                                 + 0.0257286798759461 w_k1_FDX_s004
                                 + 0.0304252267519811 w_k1_FDX_s005
                                 + 0.036402760004639 w_k1_FDX_s006
                                 + 0.0550718533822415 w_k1_FDX_s007
                                 + 0.00959468637885903 w_k1_FDX_s010
                                 + 0.0183730928435485 w_k1_FDX_s011
                                 + 0.0124240168547449 w_k1_FDX_s012
                                 + 0.0146983397516387 w_k1_FDX_s013
                                 + 0.00922127810606176 w_k1_FDX_s014
                                 + 0.00923879247220204 w_k1_FDX_s015
                                 + 0.0257494640076709 w_k1_FDX_s016
                                 + 0.010856375248496 w_k1_FDX_s017
                                 + 0.0546367588967225 w_k1_FDX_s100
                                 + 0.0354813358809272 w_k1_FDX_s101
                                 + 0.0252258963496268 w_k1_FDX_s102
                                 + 0.0138890358530561 w_k1_FDX_s103
                                 + 0.0265158449177436 w_k1_FDX_s104
                                 + 0.0121839756496814 w_k1_FDX_s105
                                 + 0.0183562686153084 w_k1_FDX_s106
                                 + 0.041042117753257 w_k1_FDX_s107
                                 + 0.0343307228856118 w_k1_FDX_s110
                                 + 0.0418415184598033 w_k1_FDX_s111
                                 + 0.0163333185835516 w_k1_FDX_s112
                                 + 0.0293997820157896 w_k1_FDX_s113
                                 + 0.0527855570390414 w_k1_FDX_s114
                                 + 0.0398080831777506 w_k1_FDX_s115
                                 + 0.0337889052448014 w_k1_FDX_s116
                                 + 0.0132940772688859 w_k1_FDX_s117
                                 + 0.0936996234437163 w_k1_IBM_s000
                                 + 0.0732741302944343 w_k1_IBM_s001
                                 + 0.0724821120695164 w_k1_IBM_s002
                                 + 0.0265318764427904 w_k1_IBM_s003
                                 + 0.0270220159146669 w_k1_IBM_s004
                                 + 0.0296165005866031 w_k1_IBM_s005
                                 + 0.0352660307317641 w_k1_IBM_s006
                                 + 0.0572066558635781 w_k1_IBM_s007
                                 + 0.00958328777091441 w_k1_IBM_s010
                                 + 0.0191859314096688 w_k1_IBM_s011
                                 + 0.0137640508533369 w_k1_IBM_s012
                                 + 0.0136930632562345 w_k1_IBM_s013
                                 + 0.00982977444934292 w_k1_IBM_s014
                                 + 0.00981443416107304 w_k1_IBM_s015
                                 + 0.0278243844335794 w_k1_IBM_s016
                                 + 0.0111434616907123 w_k1_IBM_s017
                                 + 0.0587542421711624 w_k1_IBM_s100
                                 + 0.0351084652784726 w_k1_IBM_s101
                                 + 0.024991970600629 w_k1_IBM_s102
                                 + 0.0141457083547497 w_k1_IBM_s103
                                 + 0.0255262980336836 w_k1_IBM_s104
                                 + 0.0128042245396352 w_k1_IBM_s105
                                 + 0.0188292419611628 w_k1_IBM_s106
                                 + 0.0429813923482608 w_k1_IBM_s107
                                 + 0.0324047171574664 w_k1_IBM_s110
                                 + 0.0422863770287738 w_k1_IBM_s111
                                 + 0.0157723693272296 w_k1_IBM_s112
                                 + 0.0323431947318117 w_k1_IBM_s113
                                 + 0.055961549491705 w_k1_IBM_s114
                                 + 0.0402169146394991 w_k1_IBM_s115
                                 + 0.0346060735070815 w_k1_IBM_s116
                                 + 0.0133279920262316 w_k1_IBM_s117
                                 + 0.0920645891112983 w_k1_INTC_s000
                                 + 0.0760391661711083 w_k1_INTC_s001
                                 + 0.0708059019173005 w_k1_INTC_s002
                                 + 0.0275721880464396 w_k1_INTC_s003
                                 + 0.0231311331753986 w_k1_INTC_s004
                                 + 0.027630521751066 w_k1_INTC_s005
                                 + 0.0362357357653864 w_k1_INTC_s006
                                 + 0.0479354466303336 w_k1_INTC_s007
                                 + 0.00978950974822386 w_k1_INTC_s010
                                 + 0.0196062926590684 w_k1_INTC_s011
                                 + 0.0138354315821842 w_k1_INTC_s012
                                 + 0.0144759967139098 w_k1_INTC_s013
                                 + 0.00929255891664419 w_k1_INTC_s014
                                 + 0.00877408964761816 w_k1_INTC_s015
                                 + 0.0275698011221308 w_k1_INTC_s016
                                 + 0.0106546326805787 w_k1_INTC_s017
                                 + 0.0536376078774628 w_k1_INTC_s100
                                 + 0.0359175848380175 w_k1_INTC_s101
                                 + 0.03059772646647 w_k1_INTC_s102
                                 + 0.0141322996022169 w_k1_INTC_s103
                                 + 0.0270050547870116 w_k1_INTC_s104
                                 + 0.0137913039432023 w_k1_INTC_s105
                                 + 0.015389686156932 w_k1_INTC_s106
                                 + 0.0407492245308122 w_k1_INTC_s107
                                 + 0.0379349557426613 w_k1_INTC_s110
                                 + 0.0398200374275402 w_k1_INTC_s111
                                 + 0.0174164767575056 w_k1_INTC_s112
                                 + 0.0272091088964998 w_k1_INTC_s113
                                 + 0.0516013771358348 w_k1_INTC_s114
                                 + 0.0390628609116212 w_k1_INTC_s115
                                 + 0.0336757341057707 w_k1_INTC_s116
                                 + 0.0121796085228063 w_k1_INTC_s117
                                 + 0.0919118273264763 w_k1_KO_s000
                                 + 0.0761826015957033 w_k1_KO_s001
                                 + 0.0685838861229235 w_k1_KO_s002
                                 + 0.0258158379865199 w_k1_KO_s003
                                 + 0.0251998019250066 w_k1_KO_s004
                                 + 0.0296848736785537 w_k1_KO_s005
                                 + 0.0372405134447885 w_k1_KO_s006
                                 + 0.0578838284634359 w_k1_KO_s007
                                 + 0.00859097236415238 w_k1_KO_s010
                                 + 0.0182634557619714 w_k1_KO_s011
                                 + 0.0154655512421852 w_k1_KO_s012
                                 + 0.013409459569536 w_k1_KO_s013
                                 + 0.0091123900157104 w_k1_KO_s014
                                 + 0.00862840435363877 w_k1_KO_s015
                                 + 0.028298126008388 w_k1_KO_s016
                                 + 0.0114223246552286 w_k1_KO_s017
                                 + 0.0556204252501155 w_k1_KO_s100
                                 + 0.0368941757692214 w_k1_KO_s101
                                 + 0.0265780390814136 w_k1_KO_s102
                                 + 0.0125875895731982 w_k1_KO_s103
                                 + 0.0284135754521574 w_k1_KO_s104
                                 + 0.0125853954985695 w_k1_KO_s105
                                 + 0.0193289738925273 w_k1_KO_s106
                                 + 0.0408467072277197 w_k1_KO_s107
                                 + 0.0304519046922537 w_k1_KO_s110
                                 + 0.0374026462822719 w_k1_KO_s111
                                 + 0.017083342605379 w_k1_KO_s112
                                 + 0.0300551038221252 w_k1_KO_s113
                                 + 0.0563654379242227 w_k1_KO_s114
                                 + 0.0381973837689477 w_k1_KO_s115
                                 + 0.0354148515000541 w_k1_KO_s116
                                 + 0.0135123852612516 w_k1_KO_s117
                                 + 0.0904500855707954 w_k1_MAT_s000
                                 + 0.0635271555447442 w_k1_MAT_s001
                                 + 0.0932813506601494 w_k1_MAT_s002
                                 + 0.0273276980720804 w_k1_MAT_s003
                                 + 0.0256911861691784 w_k1_MAT_s004
                                 + 0.0362032429838971 w_k1_MAT_s005
                                 + 0.0413630399180588 w_k1_MAT_s006
                                 + 0.0600343263435535 w_k1_MAT_s007
                                 + 0.00952029976162134 w_k1_MAT_s010
                                 + 0.0151484291096153 w_k1_MAT_s011
                                 + 0.0121236762652243 w_k1_MAT_s012
                                 + 0.0150301383323637 w_k1_MAT_s013
                                 + 0.010149101361433 w_k1_MAT_s014
                                 + 0.00922336463889823 w_k1_MAT_s015
                                 + 0.0251666570368082 w_k1_MAT_s016
                                 + 0.0124721607599304 w_k1_MAT_s017
                                 + 0.0616184492799197 w_k1_MAT_s100
                                 + 0.0335815795088536 w_k1_MAT_s101
                                 + 0.0231463224996634 w_k1_MAT_s102
                                 + 0.0166156421600919 w_k1_MAT_s103
                                 + 0.0214285771289071 w_k1_MAT_s104
                                 + 0.0136856641725363 w_k1_MAT_s105
                                 + 0.014760612264225 w_k1_MAT_s106
                                 + 0.0538131305600572 w_k1_MAT_s107
                                 + 0.0325272586253323 w_k1_MAT_s110
                                 + 0.0392951853717877 w_k1_MAT_s111
                                 + 0.0168776833751584 w_k1_MAT_s112
                                 + 0.0348313700650601 w_k1_MAT_s113
                                 + 0.0561683941642221 w_k1_MAT_s114
                                 + 0.0482616002774448 w_k1_MAT_s115
                                 + 0.0292505932275749 w_k1_MAT_s116
                                 + 0.0129126437843881 w_k1_MAT_s117
                                 + 0.0936073055998764 w_k1_MSFT_s000
                                 + 0.07173110563876 w_k1_MSFT_s001
                                 + 0.0744277771961534 w_k1_MSFT_s002
                                 + 0.0255122209392429 w_k1_MSFT_s003
                                 + 0.0261147929752032 w_k1_MSFT_s004
                                 + 0.0304738067076685 w_k1_MSFT_s005
                                 + 0.0377068877399309 w_k1_MSFT_s006
                                 + 0.0574330075188677 w_k1_MSFT_s007
                                 + 0.0087414370660946 w_k1_MSFT_s010
                                 + 0.019667542169385 w_k1_MSFT_s011
                                 + 0.0126412653816733 w_k1_MSFT_s012
                                 + 0.0140537698002466 w_k1_MSFT_s013
                                 + 0.00905819465703682 w_k1_MSFT_s014
                                 + 0.00906960682526847 w_k1_MSFT_s015
                                 + 0.0268357378911156 w_k1_MSFT_s016
                                 + 0.0115152185573851 w_k1_MSFT_s017
                                 + 0.0564721325056048 w_k1_MSFT_s100
                                 + 0.03444897850007 w_k1_MSFT_s101
                                 + 0.0263088410394981 w_k1_MSFT_s102
                                 + 0.0138119051625873 w_k1_MSFT_s103
                                 + 0.0259152256930814 w_k1_MSFT_s104
                                 + 0.0121167829602329 w_k1_MSFT_s105
                                 + 0.0170125765514075 w_k1_MSFT_s106
                                 + 0.0422681802830247 w_k1_MSFT_s107
                                 + 0.0332472789758737 w_k1_MSFT_s110
                                 + 0.0387838385567839 w_k1_MSFT_s111
                                 + 0.0161156056494934 w_k1_MSFT_s112
                                 + 0.0302405824005403 w_k1_MSFT_s113
                                 + 0.0538182289687607 w_k1_MSFT_s114
                                 + 0.0365849670635888 w_k1_MSFT_s115
                                 + 0.0364978270912901 w_k1_MSFT_s116
                                 + 0.012916181086216 w_k1_MSFT_s117
                                 + 0.0893290191369428 w_k1_PG_s000
                                 + 0.0773812054295405 w_k1_PG_s001
                                 + 0.0685271465406072 w_k1_PG_s002
                                 + 0.027265024854753 w_k1_PG_s003
                                 + 0.0258528486545465 w_k1_PG_s004
                                 + 0.0281901101142833 w_k1_PG_s005
                                 + 0.0365760017647841 w_k1_PG_s006
                                 + 0.0602771257751324 w_k1_PG_s007
                                 + 0.00863512475301076 w_k1_PG_s010
                                 + 0.0181007351779227 w_k1_PG_s011
                                 + 0.0132613497035185 w_k1_PG_s012
                                 + 0.0133987314091078 w_k1_PG_s013
                                 + 0.00938670833113239 w_k1_PG_s014
                                 + 0.00866480529742553 w_k1_PG_s015
                                 + 0.0262579905029044 w_k1_PG_s016
                                 + 0.00974495043157396 w_k1_PG_s017
                                 + 0.0557953513475922 w_k1_PG_s100
                                 + 0.0349401587303177 w_k1_PG_s101
                                 + 0.0232370737666662 w_k1_PG_s102
                                 + 0.0137033388827506 w_k1_PG_s103
                                 + 0.0267262189575872 w_k1_PG_s104
                                 + 0.0129986792966842 w_k1_PG_s105
                                 + 0.018872577315375 w_k1_PG_s106
                                 + 0.0397494175426919 w_k1_PG_s107
                                 + 0.0319061288550898 w_k1_PG_s110
                                 + 0.0408420497319905 w_k1_PG_s111
                                 + 0.0164967951845701 w_k1_PG_s112
                                 + 0.032328423210917 w_k1_PG_s113
                                 + 0.0539907105327552 w_k1_PG_s114
                                 + 0.0390313992852383 w_k1_PG_s115
                                 + 0.033760219099319 w_k1_PG_s116
                                 + 0.0132357273820193 w_k1_PG_s117 >= 1.16
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
