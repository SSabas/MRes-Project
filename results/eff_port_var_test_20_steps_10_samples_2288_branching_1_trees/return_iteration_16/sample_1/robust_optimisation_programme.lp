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
 w_k1_ADM_s0_return_constraint:  0.996608 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.13386 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.19991 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.855742 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.829141 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.10099 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.973281 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.842129 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.11732 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.03829 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.909584 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.965218 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.13137 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.710353 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.09293 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.02017 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.08226 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.822612 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.925844 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.938508 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.923045 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.68186 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.752193 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.13921 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.966722 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.923884 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.803899 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.28082 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.03356 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.996307 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.910394 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.944205 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.868197 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.20653 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.10024 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.15244 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.823996 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.794913 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.985608 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.21846 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.16208 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.794487 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.737821 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.23485 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.03351 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.899604 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.988094 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.62829 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.00698 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.07651 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.57558 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.697861 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.47583 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.11583 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.878401 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.825009 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.03419 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.07991 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.677654 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.497662 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.696275 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.12048 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.990651 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.866328 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.71877 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.44343 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.0188 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.23105 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.676575 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.01923 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.735773 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.29333 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.01864 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.44458 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.917768 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.843155 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.950245 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.29771 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.33227 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.722978 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.691426 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.27006 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.12123 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.744266 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.08079 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.47552 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.784701 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.02769 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.57941 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.558981 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.69152 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.21647 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.864064 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.764193 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.905026 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.34825 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.886026 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.505398 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.814135 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.27129 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.968561 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.78188 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.704556 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.47116 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.973435 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.924309 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.747252 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.964716 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.634255 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.37019 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.06389 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.58256 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.735502 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.931016 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.986965 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.10115 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.17299 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.801715 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.788791 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.15708 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.06002 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.808739 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.24595 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.16664 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.94686 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.906908 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.35277 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.785605 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.37012 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.17081 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.05259 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.843988 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.918156 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.01639 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.836863 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.650753 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.785551 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.29313 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.02772 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.914883 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.817315 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.30343 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.946768 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.03929 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.876234 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.12299 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.734022 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.16021 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.02913 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.48421 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.793903 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.83288 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.935007 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.40934 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.08431 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.919072 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.868745 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.14758 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.05706 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.11698 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.643292 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.84965 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.849611 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.67239 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.1918 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.650424 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.18479 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.902318 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.605437 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.71993 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.2834 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.73771 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.922606 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.655184 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.12375 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.800904 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.949042 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.853229 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.667592 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.45413 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.46343 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.09979 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.619547 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.654466 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.819255 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.48039 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.00142 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.00232 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.11709 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.65161 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.0011 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.985414 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.13902 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.832393 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.984298 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.36318 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.893227 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.853345 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.20733 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.27256 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.86899 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.930712 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.27186 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.772607 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.26707 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.16796 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.02297 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.887652 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.941615 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.24393 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.928667 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.75213 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.746076 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.11565 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.962424 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.95683 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.754055 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.30263 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.933661 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.965287 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.87667 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.964971 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.768181 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.04354 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.923818 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.1619 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.87516 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.711372 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.98387 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.27866 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.11448 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.798549 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.785873 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.12336 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.10835 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.941955 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.999634 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.35136 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.82367 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.24308 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.20229 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.700778 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.30936 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.11202 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.769274 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.80037 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.933055 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.51814 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.00785 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.694332 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.04241 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.04267 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.991934 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.884338 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.723932 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.59692 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.20965 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.05487 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.84563 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.814468 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.793128 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.3537 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.024 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.21987 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.02321 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.2448 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.888427 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.42802 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.20611 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.897022 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.86479 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.12297 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.04302 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.01277 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.799427 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.83429 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.935479 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.67584 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.60557 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.691872 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.14697 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.950556 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.551128 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.69515 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.42711 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.93588 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.758704 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.604897 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.820144 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.940989 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.11129 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.693789 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.530398 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.32966 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.45356 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.04896 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.730626 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.604149 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.879129 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.41232 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.04384 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.16347 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.41978 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.69486 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.999831 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.06963 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.18376 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.849879 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.884312 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.25043 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.09172 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.776426 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.12233 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.16792 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.890955 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.976771 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.2523 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.791521 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.17592 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.14622 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.935362 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.907746 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.881831 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.09405 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.867993 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.676373 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.808996 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.21054 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.989624 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.888919 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.739566 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.27005 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.917895 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.919687 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.865233 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.920352 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.741802 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.17226 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.926505 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.32562 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.83009 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.864081 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.00866 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.10569 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.06456 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.808458 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.860933 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.13712 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.0811 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.896658 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.01147 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.16361 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.92738 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.1601 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.22382 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.801976 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.24315 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.06547 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.00659 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.873371 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.02547 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.22102 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.931979 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.81984 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.840897 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.0826 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.06144 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.929529 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.809469 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.14475 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.06203 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.0494 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.834715 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.931368 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.813433 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.14664 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.957836 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.21534 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.889971 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.05302 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.39699856002299 z_k1_s0000
                                 - 1.1539946652047 z_k1_s0001
                                 - 1.09999693930531 z_k1_s0002
                                 - 0.652995612086044 z_k1_s0003
                                 - 0.521998715233484 z_k1_s0004
                                 - 0.509998469169153 z_k1_s0005
                                 - 0.562998377590064 z_k1_s0006
                                 - 1.24899328305707 z_k1_s0007
                                 - 0.297999960545817 z_k1_s0010
                                 - 0.97699776098997 z_k1_s0011
                                 - 0.739999586268462 z_k1_s0012
                                 - 1.00099809725237 z_k1_s0013
                                 - 0.522999731434307 z_k1_s0014
                                 - 1.45699860524404 z_k1_s0015
                                 - 0.378999478158086 z_k1_s0016
                                 - 0.505999542256723 z_k1_s0017
                                 - 0.261999553198953 z_k1_s0020
                                 - 0.282999735926809 z_k1_s0021
                                 - 0.497000211231653 z_k1_s0022
                                 - 1.18399666962763 z_k1_s0023
                                 - 1.18999883726121 z_k1_s0024
                                 - 0.345999296911753 z_k1_s0025
                                 - 0.503999449475419 z_k1_s0026
                                 - 0.669996898091658 z_k1_s0027
                                 - 0.483000280987285 z_k1_s0030
                                 - 0.50199907352247 z_k1_s0031
                                 - 0.124999992156094 z_k1_s0032
                                 - 0.173000154520108 z_k1_s0033
                                 - 0.228000102428767 z_k1_s0034
                                 - 0.204000142846815 z_k1_s0035
                                 - 0.302001031673594 z_k1_s0036
                                 - 0.414999098437837 z_k1_s0037
                                 - 1.15399620154984 z_k1_s0040
                                 - 1.10599380250225 z_k1_s0041
                                 - 1.58499701286979 z_k1_s0042
                                 - 1.39399961880371 z_k1_s0043
                                 - 0.879998233290682 z_k1_s0044
                                 - 0.469998900453267 z_k1_s0045
                                 - 1.13999693149165 z_k1_s0046
                                 - 0.848998256536196 z_k1_s0047
                                 - 0.543999984446199 z_k1_s0050
                                 - 0.344000056358621 z_k1_s0051
                                 - 0.348000122438367 z_k1_s0052
                                 - 0.240999855205037 z_k1_s0053
                                 - 0.99200063357831 z_k1_s0054
                                 - 0.354999862977953 z_k1_s0055
                                 - 0.705000393556163 z_k1_s0056
                                 - 0.221999916426213 z_k1_s0057
                                 - 1.01499997521344 z_k1_s0060
                                 - 0.441001213015963 z_k1_s0061
                                 - 1.29200143401561 z_k1_s0062
                                 - 1.43800032197603 z_k1_s0063
                                 - 0.862998029796148 z_k1_s0064
                                 - 0.471000643167396 z_k1_s0065
                                 - 0.773005025052817 z_k1_s0066
                                 - 1.25799921045803 z_k1_s0067
                                 - 0.541998343252803 z_k1_s0070
                                 - 1.16199714531653 z_k1_s0071
                                 - 0.822998870566924 z_k1_s0072
                                 - 0.437999003901545 z_k1_s0073
                                 - 1.72799734258356 z_k1_s0074
                                 - 0.751999138067992 z_k1_s0075
                                 - 0.465998728157911 z_k1_s0076
                                 - 0.450999080370032 z_k1_s0077
                                 - 0.459999116765242 z_k1_s0100
                                 - 0.290999379886384 z_k1_s0101
                                 - 0.660999440987469 z_k1_s0102
                                 - 0.494997427598134 z_k1_s0103
                                 - 0.529999771423746 z_k1_s0104
                                 - 0.324999525020826 z_k1_s0105
                                 - 0.371999215592888 z_k1_s0106
                                 - 0.898998440432206 z_k1_s0107
                                 - 0.382000366060271 z_k1_s0110
                                 - 1.38699814370172 z_k1_s0111
                                 - 1.12799944829176 z_k1_s0112
                                 - 0.865997653314548 z_k1_s0113
                                 - 0.701999945117991 z_k1_s0114
                                 - 0.911003472788628 z_k1_s0115
                                 - 0.607000313705751 z_k1_s0116
                                 - 0.795997639473706 z_k1_s0117
                                 - 0.333000805428374 z_k1_s0120
                                 - 0.277000042608276 z_k1_s0121
                                 - 0.282000187092251 z_k1_s0122
                                 - 0.295000299550456 z_k1_s0123
                                 - 0.492000004416108 z_k1_s0124
                                 - 0.769001692025202 z_k1_s0125
                                 - 0.341998959898484 z_k1_s0126
                                 - 0.500001288390992 z_k1_s0127
                                 - 0.660000951272771 z_k1_s0130
                                 - 0.583999973729283 z_k1_s0131
                                 - 0.636001288682314 z_k1_s0132
                                 - 0.537003191998439 z_k1_s0133
                                 - 0.392001445401403 z_k1_s0134
                                 - 0.4420029930726 z_k1_s0135
                                 - 0.479002984401315 z_k1_s0136
                                 - 0.362001253361219 z_k1_s0137
                                 - 0.949996868829666 z_k1_s0140
                                 - 0.377999543097371 z_k1_s0141
                                 - 0.591997383865368 z_k1_s0142
                                 - 0.908999271643127 z_k1_s0143
                                 - 0.386999498547358 z_k1_s0144
                                 - 0.610000123161147 z_k1_s0145
                                 - 0.317999368698154 z_k1_s0146
                                 - 0.569997178499198 z_k1_s0147
                                 - 0.613999591147918 z_k1_s0150
                                 - 0.255000319308341 z_k1_s0151
                                 - 0.517999953656334 z_k1_s0152
                                 - 0.622000652191072 z_k1_s0153
                                 - 0.199000312521146 z_k1_s0154
                                 - 0.740999520799063 z_k1_s0155
                                 - 0.59100036003151 z_k1_s0156
                                 - 0.824999749230293 z_k1_s0157
                                 - 0.213000714433376 z_k1_s0160
                                 - 0.584002951318529 z_k1_s0161
                                 - 0.670001633544107 z_k1_s0162
                                 - 0.565002938442024 z_k1_s0163
                                 - 0.354001203149892 z_k1_s0164
                                 - 0.481002488355021 z_k1_s0165
                                 - 0.489002886936007 z_k1_s0166
                                 - 0.379003008967473 z_k1_s0167
                                 - 0.294000276861292 z_k1_s0170
                                 - 0.392001167149255 z_k1_s0171
                                 - 0.512000547295927 z_k1_s0172
                                 - 0.288999205437324 z_k1_s0173
                                 - 0.323999523397311 z_k1_s0174
                                 - 0.131000061477584 z_k1_s0175
                                 - 0.20400014325064 z_k1_s0176
                                 - 0.247999516970653 z_k1_s0177
                                 - 0.0910003954135607 z_k1_s1000
                                 - 0.109000026330158 z_k1_s1001
                                 - 0.141000308849331 z_k1_s1002
                                 - 0.0879998169269307 z_k1_s1003
                                 - 0.0420000348061364 z_k1_s1004
                                 - 0.121000036274768 z_k1_s1005
                                 - 0.0630000138091728 z_k1_s1006
                                 - 0.112999773644816 z_k1_s1007
                                 - 0.145999282377138 z_k1_s1010
                                 - 0.248999137341154 z_k1_s1011
                                 - 0.0919998351795757 z_k1_s1012
                                 - 0.0769999040915226 z_k1_s1013
                                 - 0.0969998122089268 z_k1_s1014
                                 - 0.184000139158273 z_k1_s1015
                                 - 0.135999797117557 z_k1_s1016
                                 - 0.190999310040858 z_k1_s1017
                                 - 0.0710000701055897 z_k1_s1020
                                 - 0.0380000237522748 z_k1_s1021
                                 - 0.131000024493316 z_k1_s1022
                                 - 0.0820004114236611 z_k1_s1023
                                 - 0.152999804728628 z_k1_s1024
                                 - 0.0400000447918864 z_k1_s1025
                                 - 0.0960004083008041 z_k1_s1026
                                 - 0.141000129691374 z_k1_s1027
                                 - 0.094000416852748 z_k1_s1030
                                 - 0.163001081788876 z_k1_s1031
                                 - 0.230001238449226 z_k1_s1032
                                 - 0.0470002573266007 z_k1_s1033
                                 - 0.0660003426245014 z_k1_s1034
                                 - 0.0660003426245014 z_k1_s1035
                                 - 0.0979999663975026 z_k1_s1036
                                 - 0.214001084261139 z_k1_s1037
                                 - 0.0800003817407753 z_k1_s1040
                                 - 0.124000518437901 z_k1_s1041
                                 - 0.276000857463788 z_k1_s1042
                                 - 0.373001491820182 z_k1_s1043
                                 - 0.0780003755272696 z_k1_s1044
                                 - 0.145000783480531 z_k1_s1045
                                 - 0.119000436303863 z_k1_s1046
                                 - 0.137000758626508 z_k1_s1047
                                 - 0.115000425156551 z_k1_s1050
                                 - 0.131000465317396 z_k1_s1051
                                 - 0.253000908044317 z_k1_s1052
                                 - 0.185000464359768 z_k1_s1053
                                 - 0.174001255752062 z_k1_s1054
                                 - 0.177000171278388 z_k1_s1055
                                 - 0.126000452767132 z_k1_s1056
                                 - 0.204001331053646 z_k1_s1057
                                 - 0.194999528526643 z_k1_s1060
                                 - 0.165999144305355 z_k1_s1061
                                 - 0.066999926898776 z_k1_s1062
                                 - 0.159000069905717 z_k1_s1063
                                 - 0.0949998196923995 z_k1_s1064
                                 - 0.106999810432474 z_k1_s1065
                                 - 0.258000186111063 z_k1_s1066
                                 - 0.236999110917431 z_k1_s1067
                                 - 0.101999900475989 z_k1_s1070
                                 - 0.0760002265512218 z_k1_s1071
                                 - 0.045000099763168 z_k1_s1072
                                 - 0.0430000549380732 z_k1_s1073
                                 - 0.122000348726937 z_k1_s1074
                                 - 0.0460000653756238 z_k1_s1075
                                 - 0.0600002087510128 z_k1_s1076
                                 - 0.0740002953263102 z_k1_s1077
                                 - 0.180000148222726 z_k1_s1100
                                 - 0.376000149620674 z_k1_s1101
                                 - 0.137000128813977 z_k1_s1102
                                 - 0.180000148222726 z_k1_s1103
                                 - 0.129000058226248 z_k1_s1104
                                 - 0.164999655870437 z_k1_s1105
                                 - 0.121000131638637 z_k1_s1106
                                 - 0.152000765166385 z_k1_s1107
                                 - 0.0960001918706865 z_k1_s1110
                                 - 0.192000181741171 z_k1_s1111
                                 - 0.178000153760029 z_k1_s1112
                                 - 0.196000189735783 z_k1_s1113
                                 - 0.312999312577073 z_k1_s1114
                                 - 0.426000245425566 z_k1_s1115
                                 - 0.23700077668106 z_k1_s1116
                                 - 0.382000561485238 z_k1_s1117
                                 - 0.200000646504223 z_k1_s1120
                                 - 0.360999042742178 z_k1_s1121
                                 - 0.165999844999174 z_k1_s1122
                                 - 0.403998737141016 z_k1_s1123
                                 - 0.299000126724626 z_k1_s1124
                                 - 0.125999913298138 z_k1_s1125
                                 - 0.15099984591131 z_k1_s1126
                                 - 0.269000523748514 z_k1_s1127
                                 - 0.163999931891512 z_k1_s1130
                                 - 0.139999907263504 z_k1_s1131
                                 - 0.21999904375736 z_k1_s1132
                                 - 0.450999546751798 z_k1_s1133
                                 - 0.131999958154137 z_k1_s1134
                                 - 0.250999932518088 z_k1_s1135
                                 - 0.174999972729334 z_k1_s1136
                                 - 0.240000246280081 z_k1_s1137
                                 - 0.155999680120726 z_k1_s1140
                                 - 0.256999650078621 z_k1_s1141
                                 - 0.381999845446783 z_k1_s1142
                                 - 0.228998854717834 z_k1_s1143
                                 - 0.163999692624289 z_k1_s1144
                                 - 0.207999197695143 z_k1_s1145
                                 - 0.355998301613568 z_k1_s1146
                                 - 0.126999634795313 z_k1_s1147
                                 - 0.382999960426224 z_k1_s1150
                                 - 0.17200022903705 z_k1_s1151
                                 - 0.206999176482035 z_k1_s1152
                                 - 0.227999557747737 z_k1_s1153
                                 - 0.266999060099861 z_k1_s1154
                                 - 0.201999326847037 z_k1_s1155
                                 - 0.143999889482566 z_k1_s1156
                                 - 0.0849999757917403 z_k1_s1157
                                 - 0.269000464762356 z_k1_s1160
                                 - 0.0990002161230526 z_k1_s1161
                                 - 0.227000980153973 z_k1_s1162
                                 - 0.179000627742226 z_k1_s1163
                                 - 0.199000005745352 z_k1_s1164
                                 - 0.0550001347128407 z_k1_s1165
                                 - 0.0800001480185671 z_k1_s1166
                                 - 0.210000059047981 z_k1_s1167
                                 - 0.14099963541605 z_k1_s1170
                                 - 0.0669997946871048 z_k1_s1171
                                 - 0.1669998064286 z_k1_s1172
                                 - 0.0699998144193221 z_k1_s1173
                                 - 0.0889999393900315 z_k1_s1174
                                 - 0.0359999783869749 z_k1_s1175
                                 - 0.0359999783869749 z_k1_s1176
                                 - 0.0399999400966899 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.54152 w_k1_ADM_s000
                                 + 1.48346 w_k1_AXP_s000
                                 + 1.61068 w_k1_DIS_s000
                                 + 1.36672 w_k1_FDX_s000 + 1.1969 w_k1_IBM_s000
                                 + 1.54037 w_k1_INTC_s000
                                 + 1.33622 w_k1_KO_s000 + 1.27339 w_k1_MAT_s000
                                 + 1.277 w_k1_MSFT_s000 + 1.31004 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.991402 w_k1_ADM_s000
                                 + 0.863025 w_k1_AXP_s000
                                 + 0.930023 w_k1_DIS_s000
                                 + 0.899233 w_k1_FDX_s000
                                 + 1.00093 w_k1_IBM_s000
                                 + 0.988825 w_k1_INTC_s000
                                 + 0.998069 w_k1_KO_s000
                                 + 0.840502 w_k1_MAT_s000
                                 + 0.828248 w_k1_MSFT_s000
                                 + 0.912252 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.11541 w_k1_ADM_s000
                                 + 1.00627 w_k1_AXP_s000
                                 + 1.01648 w_k1_DIS_s000
                                 + 1.11938 w_k1_FDX_s000
                                 + 0.840101 w_k1_IBM_s000
                                 + 1.09228 w_k1_INTC_s000
                                 + 0.902291 w_k1_KO_s000
                                 + 0.79349 w_k1_MAT_s000
                                 + 0.891309 w_k1_MSFT_s000
                                 + 0.899068 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.14634 w_k1_ADM_s000
                                 + 1.08815 w_k1_AXP_s000
                                 + 0.977321 w_k1_DIS_s000
                                 + 0.982702 w_k1_FDX_s000
                                 + 1.40171 w_k1_IBM_s000
                                 + 1.09011 w_k1_INTC_s000
                                 + 1.08938 w_k1_KO_s000 + 1.09833 w_k1_MAT_s000
                                 + 0.847522 w_k1_MSFT_s000
                                 + 1.08347 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.05658 w_k1_ADM_s000
                                 + 0.952253 w_k1_AXP_s000
                                 + 0.697324 w_k1_DIS_s000
                                 + 0.876916 w_k1_FDX_s000
                                 + 0.697207 w_k1_IBM_s000
                                 + 0.98207 w_k1_INTC_s000
                                 + 0.752077 w_k1_KO_s000
                                 + 0.716527 w_k1_MAT_s000
                                 + 0.824271 w_k1_MSFT_s000
                                 + 0.854126 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.782974 w_k1_ADM_s000
                                 + 0.815829 w_k1_AXP_s000
                                 + 0.75789 w_k1_DIS_s000
                                 + 0.737958 w_k1_FDX_s000
                                 + 1.22151 w_k1_IBM_s000
                                 + 1.04166 w_k1_INTC_s000
                                 + 0.955463 w_k1_KO_s000 + 1.0447 w_k1_MAT_s000
                                 + 0.794387 w_k1_MSFT_s000
                                 + 0.883979 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.798165 w_k1_ADM_s000
                                 + 0.671484 w_k1_AXP_s000
                                 + 0.681835 w_k1_DIS_s000
                                 + 0.768442 w_k1_FDX_s000
                                 + 0.704056 w_k1_IBM_s000
                                 + 0.95897 w_k1_INTC_s000
                                 + 0.817265 w_k1_KO_s000
                                 + 0.693106 w_k1_MAT_s000
                                 + 0.742122 w_k1_MSFT_s000
                                 + 0.808839 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.15329 w_k1_ADM_s000
                                 + 1.05848 w_k1_AXP_s000 + 1.3203 w_k1_DIS_s000
                                 + 1.22571 w_k1_FDX_s000
                                 + 1.19889 w_k1_IBM_s000
                                 + 1.14418 w_k1_INTC_s000
                                 + 1.18897 w_k1_KO_s000
                                 + 0.969129 w_k1_MAT_s000
                                 + 1.09657 w_k1_MSFT_s000
                                 + 1.10883 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.885902 w_k1_ADM_s001
                                 + 0.900202 w_k1_AXP_s001
                                 + 0.718802 w_k1_DIS_s001
                                 + 0.747004 w_k1_FDX_s001
                                 + 1.34453 w_k1_IBM_s001
                                 + 0.886841 w_k1_INTC_s001
                                 + 0.983292 w_k1_KO_s001
                                 + 1.25405 w_k1_MAT_s001
                                 + 0.876686 w_k1_MSFT_s001
                                 + 0.977702 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.824824 w_k1_ADM_s001
                                 + 0.763162 w_k1_AXP_s001
                                 + 0.80394 w_k1_DIS_s001
                                 + 0.840119 w_k1_FDX_s001
                                 + 1.00761 w_k1_IBM_s001
                                 + 0.824065 w_k1_INTC_s001
                                 + 0.896854 w_k1_KO_s001
                                 + 0.802489 w_k1_MAT_s001
                                 + 0.943341 w_k1_MSFT_s001
                                 + 0.866869 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.10532 w_k1_ADM_s001
                                 + 1.47968 w_k1_AXP_s001
                                 + 1.40694 w_k1_DIS_s001
                                 + 1.34775 w_k1_FDX_s001
                                 + 1.24316 w_k1_IBM_s001
                                 + 1.07065 w_k1_INTC_s001
                                 + 1.29616 w_k1_KO_s001 + 1.44493 w_k1_MAT_s001
                                 + 1.23093 w_k1_MSFT_s001
                                 + 1.28617 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.976631 w_k1_ADM_s001
                                 + 1.03748 w_k1_AXP_s001
                                 + 1.20303 w_k1_DIS_s001
                                 + 1.03985 w_k1_FDX_s001
                                 + 1.20348 w_k1_IBM_s001
                                 + 1.12025 w_k1_INTC_s001
                                 + 1.09707 w_k1_KO_s001 + 1.18317 w_k1_MAT_s001
                                 + 1.06571 w_k1_MSFT_s001
                                 + 1.03466 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.926871 w_k1_ADM_s001
                                 + 1.22715 w_k1_AXP_s001
                                 + 1.33379 w_k1_DIS_s001
                                 + 1.09531 w_k1_FDX_s001
                                 + 1.64054 w_k1_IBM_s001
                                 + 1.24879 w_k1_INTC_s001
                                 + 1.32983 w_k1_KO_s001 + 1.46366 w_k1_MAT_s001
                                 + 1.19474 w_k1_MSFT_s001
                                 + 1.11707 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.07139 w_k1_ADM_s001
                                 + 1.01457 w_k1_AXP_s001
                                 + 1.01235 w_k1_DIS_s001
                                 + 1.12412 w_k1_FDX_s001
                                 + 0.86056 w_k1_IBM_s001
                                 + 0.925545 w_k1_INTC_s001
                                 + 0.840566 w_k1_KO_s001
                                 + 1.18919 w_k1_MAT_s001
                                 + 1.06764 w_k1_MSFT_s001 + 1.0558 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.825208 w_k1_ADM_s001
                                 + 0.689531 w_k1_AXP_s001
                                 + 0.584343 w_k1_DIS_s001
                                 + 0.784902 w_k1_FDX_s001
                                 + 0.846512 w_k1_IBM_s001
                                 + 0.637447 w_k1_INTC_s001
                                 + 0.767484 w_k1_KO_s001
                                 + 0.995347 w_k1_MAT_s001
                                 + 0.711134 w_k1_MSFT_s001
                                 + 0.883088 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.781892 w_k1_ADM_s001
                                 + 0.638744 w_k1_AXP_s001
                                 + 0.625679 w_k1_DIS_s001
                                 + 0.775935 w_k1_FDX_s001
                                 + 0.624305 w_k1_IBM_s001
                                 + 0.787858 w_k1_INTC_s001
                                 + 0.712802 w_k1_KO_s001
                                 + 0.545525 w_k1_MAT_s001
                                 + 0.826899 w_k1_MSFT_s001
                                 + 0.794685 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.9856 w_k1_ADM_s002
                                 + 1.41018 w_k1_AXP_s002
                                 + 1.64243 w_k1_DIS_s002
                                 + 1.20301 w_k1_FDX_s002
                                 + 1.45763 w_k1_IBM_s002
                                 + 1.18741 w_k1_INTC_s002
                                 + 1.20688 w_k1_KO_s002 + 1.21306 w_k1_MAT_s002
                                 + 1.25426 w_k1_MSFT_s002
                                 + 1.15002 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.983521 w_k1_ADM_s002
                                 + 1.08495 w_k1_AXP_s002
                                 + 1.82876 w_k1_DIS_s002
                                 + 1.16784 w_k1_FDX_s002
                                 + 1.49328 w_k1_IBM_s002
                                 + 1.08293 w_k1_INTC_s002
                                 + 1.44961 w_k1_KO_s002 + 1.25615 w_k1_MAT_s002
                                 + 1.41279 w_k1_MSFT_s002
                                 + 1.30767 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.19415 w_k1_ADM_s002
                                 + 1.08104 w_k1_AXP_s002
                                 + 1.23494 w_k1_DIS_s002 + 1.1159 w_k1_FDX_s002
                                 + 1.1101 w_k1_IBM_s002
                                 + 1.01047 w_k1_INTC_s002
                                 + 0.999424 w_k1_KO_s002
                                 + 0.747978 w_k1_MAT_s002
                                 + 1.08735 w_k1_MSFT_s002
                                 + 1.02455 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.825069 w_k1_ADM_s002
                                 + 0.920695 w_k1_AXP_s002
                                 + 0.836334 w_k1_DIS_s002
                                 + 0.95216 w_k1_FDX_s002
                                 + 0.720946 w_k1_IBM_s002
                                 + 0.900125 w_k1_INTC_s002
                                 + 0.640265 w_k1_KO_s002
                                 + 0.678785 w_k1_MAT_s002
                                 + 0.878888 w_k1_MSFT_s002
                                 + 0.90774 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.729469 w_k1_ADM_s002
                                 + 0.68302 w_k1_AXP_s002
                                 + 0.798662 w_k1_DIS_s002
                                 + 0.74423 w_k1_FDX_s002
                                 + 0.920207 w_k1_IBM_s002
                                 + 0.727209 w_k1_INTC_s002
                                 + 0.89045 w_k1_KO_s002 + 0.78498 w_k1_MAT_s002
                                 + 0.786217 w_k1_MSFT_s002
                                 + 0.898479 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.0569 w_k1_ADM_s002
                                 + 1.044 w_k1_AXP_s002 + 1.07417 w_k1_DIS_s002
                                 + 0.945219 w_k1_FDX_s002
                                 + 1.0009 w_k1_IBM_s002
                                 + 0.863343 w_k1_INTC_s002
                                 + 0.945736 w_k1_KO_s002
                                 + 0.867324 w_k1_MAT_s002
                                 + 0.967118 w_k1_MSFT_s002
                                 + 1.04037 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.29111 w_k1_ADM_s002
                                 + 1.7691 w_k1_AXP_s002 + 1.53696 w_k1_DIS_s002
                                 + 1.42921 w_k1_FDX_s002
                                 + 0.994408 w_k1_IBM_s002
                                 + 1.0993 w_k1_INTC_s002 + 1.05219 w_k1_KO_s002
                                 + 1.16638 w_k1_MAT_s002
                                 + 1.25807 w_k1_MSFT_s002
                                 + 1.33022 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.849483 w_k1_ADM_s002
                                 + 1.0365 w_k1_AXP_s002 + 1.15891 w_k1_DIS_s002
                                 + 0.938361 w_k1_FDX_s002
                                 + 1.35754 w_k1_IBM_s002
                                 + 0.759425 w_k1_INTC_s002
                                 + 1.06691 w_k1_KO_s002
                                 + 0.995775 w_k1_MAT_s002
                                 + 0.955737 w_k1_MSFT_s002
                                 + 1.07338 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.0952 w_k1_ADM_s003
                                 + 1.11932 w_k1_AXP_s003
                                 + 0.998856 w_k1_DIS_s003
                                 + 0.946796 w_k1_FDX_s003
                                 + 1.42378 w_k1_IBM_s003
                                 + 0.993063 w_k1_INTC_s003
                                 + 1.26747 w_k1_KO_s003 + 1.55799 w_k1_MAT_s003
                                 + 1.09452 w_k1_MSFT_s003 + 1.0138 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.925293 w_k1_ADM_s003
                                 + 0.719872 w_k1_AXP_s003
                                 + 0.651234 w_k1_DIS_s003
                                 + 0.89919 w_k1_FDX_s003
                                 + 0.929346 w_k1_IBM_s003
                                 + 0.787224 w_k1_INTC_s003
                                 + 0.848035 w_k1_KO_s003
                                 + 0.912382 w_k1_MAT_s003
                                 + 0.896429 w_k1_MSFT_s003
                                 + 0.911741 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.726256 w_k1_ADM_s003
                                 + 0.818939 w_k1_AXP_s003
                                 + 0.628182 w_k1_DIS_s003
                                 + 0.788121 w_k1_FDX_s003
                                 + 1.15132 w_k1_IBM_s003
                                 + 0.849874 w_k1_INTC_s003
                                 + 0.983351 w_k1_KO_s003 + 1.2825 w_k1_MAT_s003
                                 + 0.825754 w_k1_MSFT_s003
                                 + 0.875818 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.34409 w_k1_ADM_s003
                                 + 1.51109 w_k1_AXP_s003
                                 + 1.81396 w_k1_DIS_s003
                                 + 1.54019 w_k1_FDX_s003
                                 + 1.50042 w_k1_IBM_s003
                                 + 1.53233 w_k1_INTC_s003
                                 + 1.78714 w_k1_KO_s003 + 1.86391 w_k1_MAT_s003
                                 + 1.35559 w_k1_MSFT_s003 + 1.3928 w_k1_PG_s003
                                 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.09081 w_k1_ADM_s003
                                 + 0.886235 w_k1_AXP_s003
                                 + 0.663879 w_k1_DIS_s003
                                 + 0.993698 w_k1_FDX_s003
                                 + 0.702113 w_k1_IBM_s003
                                 + 0.711516 w_k1_INTC_s003
                                 + 0.787138 w_k1_KO_s003
                                 + 0.947337 w_k1_MAT_s003
                                 + 0.851296 w_k1_MSFT_s003
                                 + 0.953325 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.994051 w_k1_ADM_s003
                                 + 0.674823 w_k1_AXP_s003
                                 + 0.557054 w_k1_DIS_s003
                                 + 0.801317 w_k1_FDX_s003
                                 + 0.613169 w_k1_IBM_s003
                                 + 0.769282 w_k1_INTC_s003
                                 + 0.710377 w_k1_KO_s003
                                 + 0.599822 w_k1_MAT_s003
                                 + 0.842016 w_k1_MSFT_s003
                                 + 0.794691 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.876048 w_k1_ADM_s003
                                 + 0.802793 w_k1_AXP_s003
                                 + 1.12552 w_k1_DIS_s003
                                 + 1.12589 w_k1_FDX_s003
                                 + 0.796793 w_k1_IBM_s003
                                 + 1.09264 w_k1_INTC_s003
                                 + 1.04031 w_k1_KO_s003
                                 + 0.851382 w_k1_MAT_s003
                                 + 1.08384 w_k1_MSFT_s003
                                 + 0.987512 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.13613 w_k1_ADM_s003
                                 + 1.11107 w_k1_AXP_s003
                                 + 1.23274 w_k1_DIS_s003
                                 + 1.19821 w_k1_FDX_s003
                                 + 1.08144 w_k1_IBM_s003
                                 + 0.986172 w_k1_INTC_s003
                                 + 1.27692 w_k1_KO_s003 + 1.17438 w_k1_MAT_s003
                                 + 1.022 w_k1_MSFT_s003 + 1.12781 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.944491 w_k1_ADM_s004
                                 + 1.00922 w_k1_AXP_s004
                                 + 1.13399 w_k1_DIS_s004
                                 + 1.08535 w_k1_FDX_s004 + 1.0317 w_k1_IBM_s004
                                 + 1.22097 w_k1_INTC_s004
                                 + 1.05068 w_k1_KO_s004
                                 + 0.935604 w_k1_MAT_s004
                                 + 1.05118 w_k1_MSFT_s004 + 1.1807 w_k1_PG_s004
                                 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.802672 w_k1_ADM_s004
                                 + 0.706166 w_k1_AXP_s004
                                 + 0.791377 w_k1_DIS_s004
                                 + 0.861255 w_k1_FDX_s004
                                 + 0.834634 w_k1_IBM_s004
                                 + 0.803592 w_k1_INTC_s004
                                 + 0.905804 w_k1_KO_s004
                                 + 0.723882 w_k1_MAT_s004
                                 + 0.851316 w_k1_MSFT_s004
                                 + 0.961307 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.25821 w_k1_ADM_s004
                                 + 1.25495 w_k1_AXP_s004
                                 + 1.48448 w_k1_DIS_s004
                                 + 1.26435 w_k1_FDX_s004
                                 + 0.999371 w_k1_IBM_s004
                                 + 1.37246 w_k1_INTC_s004
                                 + 1.12464 w_k1_KO_s004 + 1.00291 w_k1_MAT_s004
                                 + 1.17032 w_k1_MSFT_s004
                                 + 1.19737 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.974913 w_k1_ADM_s004
                                 + 0.780968 w_k1_AXP_s004
                                 + 0.907823 w_k1_DIS_s004
                                 + 1.01281 w_k1_FDX_s004
                                 + 0.747367 w_k1_IBM_s004
                                 + 1.16993 w_k1_INTC_s004
                                 + 0.912323 w_k1_KO_s004
                                 + 0.634682 w_k1_MAT_s004
                                 + 0.94564 w_k1_MSFT_s004
                                 + 1.00572 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.876305 w_k1_ADM_s004
                                 + 1.30436 w_k1_AXP_s004
                                 + 1.23972 w_k1_DIS_s004
                                 + 1.11176 w_k1_FDX_s004 + 1.1187 w_k1_IBM_s004
                                 + 1.09892 w_k1_INTC_s004
                                 + 1.09395 w_k1_KO_s004 + 1.17259 w_k1_MAT_s004
                                 + 0.99904 w_k1_MSFT_s004
                                 + 0.989714 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.30839 w_k1_ADM_s004
                                 + 1.70018 w_k1_AXP_s004
                                 + 1.35761 w_k1_DIS_s004
                                 + 1.31933 w_k1_FDX_s004
                                 + 1.44634 w_k1_IBM_s004
                                 + 1.31475 w_k1_INTC_s004
                                 + 1.13162 w_k1_KO_s004 + 1.67895 w_k1_MAT_s004
                                 + 1.07285 w_k1_MSFT_s004
                                 + 1.29706 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.936105 w_k1_ADM_s004
                                 + 0.876754 w_k1_AXP_s004
                                 + 1.11635 w_k1_DIS_s004
                                 + 0.906772 w_k1_FDX_s004
                                 + 1.30567 w_k1_IBM_s004
                                 + 1.00934 w_k1_INTC_s004
                                 + 1.39096 w_k1_KO_s004 + 1.18118 w_k1_MAT_s004
                                 + 1.12153 w_k1_MSFT_s004
                                 + 1.13064 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.806541 w_k1_ADM_s004
                                 + 0.621734 w_k1_AXP_s004
                                 + 0.657642 w_k1_DIS_s004
                                 + 0.733565 w_k1_FDX_s004
                                 + 0.862147 w_k1_IBM_s004
                                 + 0.84379 w_k1_INTC_s004
                                 + 0.868302 w_k1_KO_s004
                                 + 0.558365 w_k1_MAT_s004
                                 + 0.773897 w_k1_MSFT_s004
                                 + 0.837972 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.826536 w_k1_ADM_s005
                                 + 0.796303 w_k1_AXP_s005
                                 + 0.694781 w_k1_DIS_s005
                                 + 0.821748 w_k1_FDX_s005
                                 + 0.969572 w_k1_IBM_s005
                                 + 0.916766 w_k1_INTC_s005
                                 + 0.7989 w_k1_KO_s005 + 0.81206 w_k1_MAT_s005
                                 + 0.818322 w_k1_MSFT_s005
                                 + 0.806881 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.957946 w_k1_ADM_s005
                                 + 1.01969 w_k1_AXP_s005
                                 + 1.28558 w_k1_DIS_s005
                                 + 1.10644 w_k1_FDX_s005
                                 + 0.991957 w_k1_IBM_s005
                                 + 1.07116 w_k1_INTC_s005
                                 + 1.05987 w_k1_KO_s005 + 1.17991 w_k1_MAT_s005
                                 + 1.20564 w_k1_MSFT_s005
                                 + 1.03951 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.15786 w_k1_ADM_s005
                                 + 1.04873 w_k1_AXP_s005
                                 + 1.03063 w_k1_DIS_s005
                                 + 1.18288 w_k1_FDX_s005
                                 + 0.890803 w_k1_IBM_s005
                                 + 1.19301 w_k1_INTC_s005
                                 + 1.01245 w_k1_KO_s005
                                 + 0.935945 w_k1_MAT_s005
                                 + 1.0041 w_k1_MSFT_s005
                                 + 0.929832 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.708603 w_k1_ADM_s005
                                 + 0.925133 w_k1_AXP_s005
                                 + 0.62851 w_k1_DIS_s005
                                 + 0.828737 w_k1_FDX_s005
                                 + 0.717563 w_k1_IBM_s005
                                 + 0.705344 w_k1_INTC_s005
                                 + 0.716302 w_k1_KO_s005
                                 + 0.901697 w_k1_MAT_s005
                                 + 0.688065 w_k1_MSFT_s005
                                 + 0.71481 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.28913 w_k1_ADM_s005
                                 + 1.25761 w_k1_AXP_s005
                                 + 1.37364 w_k1_DIS_s005
                                 + 1.18657 w_k1_FDX_s005
                                 + 1.34445 w_k1_IBM_s005
                                 + 1.35533 w_k1_INTC_s005
                                 + 1.38443 w_k1_KO_s005 + 1.16378 w_k1_MAT_s005
                                 + 1.11485 w_k1_MSFT_s005
                                 + 1.10341 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.761127 w_k1_ADM_s005
                                 + 0.700124 w_k1_AXP_s005
                                 + 0.871174 w_k1_DIS_s005
                                 + 0.932774 w_k1_FDX_s005
                                 + 0.627088 w_k1_IBM_s005
                                 + 0.908317 w_k1_INTC_s005
                                 + 0.794892 w_k1_KO_s005
                                 + 0.428866 w_k1_MAT_s005
                                 + 0.848698 w_k1_MSFT_s005
                                 + 0.868555 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.809839 w_k1_ADM_s005
                                 + 0.898109 w_k1_AXP_s005
                                 + 0.982603 w_k1_DIS_s005
                                 + 0.938058 w_k1_FDX_s005
                                 + 1.07683 w_k1_IBM_s005
                                 + 1.05531 w_k1_INTC_s005
                                 + 1.03898 w_k1_KO_s005 + 1.0292 w_k1_MAT_s005
                                 + 0.977811 w_k1_MSFT_s005
                                 + 0.993062 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.997811 w_k1_ADM_s005
                                 + 0.904783 w_k1_AXP_s005
                                 + 1.1573 w_k1_DIS_s005 + 1.10757 w_k1_FDX_s005
                                 + 0.71697 w_k1_IBM_s005
                                 + 1.0942 w_k1_INTC_s005 + 1.00204 w_k1_KO_s005
                                 + 0.802511 w_k1_MAT_s005
                                 + 1.016 w_k1_MSFT_s005 + 0.893176 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.14913 w_k1_ADM_s006
                                 + 1.0975 w_k1_AXP_s006 + 1.17849 w_k1_DIS_s006
                                 + 1.11303 w_k1_FDX_s006
                                 + 1.13292 w_k1_IBM_s006
                                 + 1.03366 w_k1_INTC_s006
                                 + 1.26168 w_k1_KO_s006 + 1.03233 w_k1_MAT_s006
                                 + 1.08654 w_k1_MSFT_s006
                                 + 1.06318 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.18641 w_k1_ADM_s006
                                 + 0.919924 w_k1_AXP_s006
                                 + 1.40724 w_k1_DIS_s006 + 1.2703 w_k1_FDX_s006
                                 + 0.975319 w_k1_IBM_s006
                                 + 1.14675 w_k1_INTC_s006
                                 + 1.19843 w_k1_KO_s006 + 0.71352 w_k1_MAT_s006
                                 + 1.24398 w_k1_MSFT_s006
                                 + 1.11183 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.953383 w_k1_ADM_s006
                                 + 0.869361 w_k1_AXP_s006
                                 + 0.551584 w_k1_DIS_s006
                                 + 0.792446 w_k1_FDX_s006
                                 + 0.910641 w_k1_IBM_s006
                                 + 0.726741 w_k1_INTC_s006
                                 + 0.79697 w_k1_KO_s006
                                 + 0.953651 w_k1_MAT_s006
                                 + 0.709142 w_k1_MSFT_s006
                                 + 0.844416 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.1457 w_k1_ADM_s006
                                 + 1.03347 w_k1_AXP_s006
                                 + 0.977633 w_k1_DIS_s006
                                 + 0.927927 w_k1_FDX_s006
                                 + 1.46503 w_k1_IBM_s006
                                 + 1.02045 w_k1_INTC_s006
                                 + 1.03946 w_k1_KO_s006 + 1.16323 w_k1_MAT_s006
                                 + 0.842904 w_k1_MSFT_s006
                                 + 1.00635 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.27917 w_k1_ADM_s006
                                 + 1.02485 w_k1_AXP_s006
                                 + 1.00801 w_k1_DIS_s006
                                 + 1.12375 w_k1_FDX_s006
                                 + 0.706973 w_k1_IBM_s006
                                 + 0.875069 w_k1_INTC_s006
                                 + 0.92068 w_k1_KO_s006
                                 + 0.451039 w_k1_MAT_s006
                                 + 0.983788 w_k1_MSFT_s006
                                 + 0.958959 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.23712 w_k1_ADM_s006
                                 + 1.54577 w_k1_AXP_s006
                                 + 1.08418 w_k1_DIS_s006
                                 + 1.15375 w_k1_FDX_s006
                                 + 1.09943 w_k1_IBM_s006
                                 + 1.06876 w_k1_INTC_s006 + 1.1327 w_k1_KO_s006
                                 + 1.37651 w_k1_MAT_s006
                                 + 1.02334 w_k1_MSFT_s006
                                 + 0.985826 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.82387 w_k1_ADM_s006
                                 + 1.47061 w_k1_AXP_s006
                                 + 1.86629 w_k1_DIS_s006
                                 + 1.60066 w_k1_FDX_s006
                                 + 1.24743 w_k1_IBM_s006
                                 + 1.5314 w_k1_INTC_s006 + 1.70709 w_k1_KO_s006
                                 + 1.10493 w_k1_MAT_s006
                                 + 1.42204 w_k1_MSFT_s006
                                 + 1.35654 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.973572 w_k1_ADM_s006
                                 + 0.672913 w_k1_AXP_s006
                                 + 0.803535 w_k1_DIS_s006
                                 + 0.826235 w_k1_FDX_s006
                                 + 0.922589 w_k1_IBM_s006
                                 + 0.830039 w_k1_INTC_s006
                                 + 0.908029 w_k1_KO_s006 + 0.5074 w_k1_MAT_s006
                                 + 0.816913 w_k1_MSFT_s006
                                 + 0.809053 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.45004 w_k1_ADM_s007
                                 + 1.50326 w_k1_AXP_s007
                                 + 1.13132 w_k1_DIS_s007
                                 + 1.18009 w_k1_FDX_s007
                                 + 1.11926 w_k1_IBM_s007
                                 + 1.06735 w_k1_INTC_s007
                                 + 1.13465 w_k1_KO_s007 + 1.33266 w_k1_MAT_s007
                                 + 0.975369 w_k1_MSFT_s007
                                 + 1.17014 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.15226 w_k1_ADM_s007
                                 + 1.20105 w_k1_AXP_s007
                                 + 1.35872 w_k1_DIS_s007
                                 + 1.24696 w_k1_FDX_s007
                                 + 0.792719 w_k1_IBM_s007
                                 + 1.09562 w_k1_INTC_s007
                                 + 0.956169 w_k1_KO_s007
                                 + 0.538529 w_k1_MAT_s007
                                 + 1.089 w_k1_MSFT_s007 + 1.13682 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.38175 w_k1_ADM_s007
                                 + 0.98305 w_k1_AXP_s007
                                 + 1.41726 w_k1_DIS_s007
                                 + 1.09716 w_k1_FDX_s007
                                 + 1.14741 w_k1_IBM_s007
                                 + 1.29827 w_k1_INTC_s007
                                 + 1.15329 w_k1_KO_s007 + 1.03487 w_k1_MAT_s007
                                 + 1.01315 w_k1_MSFT_s007
                                 + 1.12877 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.11703 w_k1_ADM_s007
                                 + 1.19558 w_k1_AXP_s007
                                 + 1.44831 w_k1_DIS_s007
                                 + 0.958956 w_k1_FDX_s007
                                 + 1.82165 w_k1_IBM_s007
                                 + 0.869534 w_k1_INTC_s007
                                 + 1.3655 w_k1_KO_s007 + 1.71811 w_k1_MAT_s007
                                 + 0.897065 w_k1_MSFT_s007
                                 + 1.10166 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.935591 w_k1_ADM_s007
                                 + 0.905152 w_k1_AXP_s007
                                 + 0.952537 w_k1_DIS_s007
                                 + 0.881842 w_k1_FDX_s007
                                 + 0.855547 w_k1_IBM_s007
                                 + 0.852765 w_k1_INTC_s007
                                 + 0.971446 w_k1_KO_s007
                                 + 0.74179 w_k1_MAT_s007
                                 + 0.925752 w_k1_MSFT_s007
                                 + 1.00101 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.793121 w_k1_ADM_s007
                                 + 0.710493 w_k1_AXP_s007
                                 + 0.629783 w_k1_DIS_s007
                                 + 0.727673 w_k1_FDX_s007
                                 + 0.636973 w_k1_IBM_s007
                                 + 0.840311 w_k1_INTC_s007
                                 + 0.597053 w_k1_KO_s007
                                 + 0.566565 w_k1_MAT_s007
                                 + 0.732506 w_k1_MSFT_s007
                                 + 0.782782 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.8736 w_k1_ADM_s007
                                 + 0.670392 w_k1_AXP_s007
                                 + 0.780004 w_k1_DIS_s007
                                 + 0.710957 w_k1_FDX_s007
                                 + 1.18911 w_k1_IBM_s007
                                 + 0.734516 w_k1_INTC_s007
                                 + 0.911414 w_k1_KO_s007
                                 + 0.814802 w_k1_MAT_s007
                                 + 0.727956 w_k1_MSFT_s007
                                 + 0.889875 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.07877 w_k1_ADM_s007
                                 + 1.17624 w_k1_AXP_s007 + 1.0365 w_k1_DIS_s007
                                 + 0.915437 w_k1_FDX_s007
                                 + 1.33477 w_k1_IBM_s007
                                 + 0.822462 w_k1_INTC_s007
                                 + 1.11918 w_k1_KO_s007 + 1.3157 w_k1_MAT_s007
                                 + 0.849807 w_k1_MSFT_s007
                                 + 1.06939 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.844761 w_k1_ADM_s010
                                 + 0.845663 w_k1_AXP_s010
                                 + 1.05318 w_k1_DIS_s010
                                 + 0.883973 w_k1_FDX_s010
                                 + 1.38407 w_k1_IBM_s010
                                 + 0.908836 w_k1_INTC_s010
                                 + 1.25866 w_k1_KO_s010 + 1.80761 w_k1_MAT_s010
                                 + 1.09105 w_k1_MSFT_s010
                                 + 1.01782 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.23741 w_k1_ADM_s010
                                 + 1.30372 w_k1_AXP_s010 + 1.5469 w_k1_DIS_s010
                                 + 1.34902 w_k1_FDX_s010 + 1.0093 w_k1_IBM_s010
                                 + 1.41094 w_k1_INTC_s010
                                 + 1.34305 w_k1_KO_s010 + 1.45054 w_k1_MAT_s010
                                 + 1.32401 w_k1_MSFT_s010
                                 + 1.19425 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.09069 w_k1_ADM_s010
                                 + 0.742003 w_k1_AXP_s010
                                 + 0.758081 w_k1_DIS_s010
                                 + 0.939556 w_k1_FDX_s010
                                 + 0.658024 w_k1_IBM_s010
                                 + 0.968784 w_k1_INTC_s010
                                 + 0.767089 w_k1_KO_s010
                                 + 0.896006 w_k1_MAT_s010
                                 + 0.977547 w_k1_MSFT_s010
                                 + 0.975439 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.27304 w_k1_ADM_s010
                                 + 1.06049 w_k1_AXP_s010
                                 + 1.10797 w_k1_DIS_s010
                                 + 1.23746 w_k1_FDX_s010
                                 + 0.780613 w_k1_IBM_s010
                                 + 1.10751 w_k1_INTC_s010
                                 + 0.955172 w_k1_KO_s010
                                 + 0.947417 w_k1_MAT_s010
                                 + 1.1857 w_k1_MSFT_s010 + 1.07706 w_k1_PG_s010
                                 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.23049 w_k1_ADM_s010
                                 + 0.945672 w_k1_AXP_s010
                                 + 0.77318 w_k1_DIS_s010
                                 + 0.88745 w_k1_FDX_s010 + 1.1092 w_k1_IBM_s010
                                 + 0.854642 w_k1_INTC_s010
                                 + 0.950167 w_k1_KO_s010
                                 + 1.30378 w_k1_MAT_s010
                                 + 0.955596 w_k1_MSFT_s010
                                 + 0.998804 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.29328 w_k1_ADM_s010
                                 + 1.01866 w_k1_AXP_s010
                                 + 1.19286 w_k1_DIS_s010
                                 + 1.09932 w_k1_FDX_s010
                                 + 1.20747 w_k1_IBM_s010
                                 + 1.0826 w_k1_INTC_s010 + 1.18942 w_k1_KO_s010
                                 + 1.42596 w_k1_MAT_s010
                                 + 1.22203 w_k1_MSFT_s010
                                 + 1.08446 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.44345 w_k1_ADM_s010
                                 + 1.46291 w_k1_AXP_s010
                                 + 1.55397 w_k1_DIS_s010
                                 + 1.34071 w_k1_FDX_s010
                                 + 1.69673 w_k1_IBM_s010
                                 + 1.06473 w_k1_INTC_s010
                                 + 1.42865 w_k1_KO_s010 + 2.76825 w_k1_MAT_s010
                                 + 1.25099 w_k1_MSFT_s010
                                 + 1.31701 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.914478 w_k1_ADM_s010
                                 + 0.590887 w_k1_AXP_s010
                                 + 0.755041 w_k1_DIS_s010
                                 + 0.80669 w_k1_FDX_s010
                                 + 0.840078 w_k1_IBM_s010
                                 + 0.820033 w_k1_INTC_s010
                                 + 0.966868 w_k1_KO_s010
                                 + 0.835028 w_k1_MAT_s010
                                 + 0.993791 w_k1_MSFT_s010
                                 + 0.949671 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.855786 w_k1_ADM_s011
                                 + 0.767086 w_k1_AXP_s011
                                 + 0.629422 w_k1_DIS_s011
                                 + 0.681942 w_k1_FDX_s011
                                 + 0.954157 w_k1_IBM_s011
                                 + 0.642053 w_k1_INTC_s011
                                 + 0.782883 w_k1_KO_s011
                                 + 1.16523 w_k1_MAT_s011
                                 + 0.712119 w_k1_MSFT_s011
                                 + 0.85974 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.07475 w_k1_ADM_s011
                                 + 0.790038 w_k1_AXP_s011
                                 + 0.754726 w_k1_DIS_s011
                                 + 0.900571 w_k1_FDX_s011
                                 + 0.692511 w_k1_IBM_s011
                                 + 0.823562 w_k1_INTC_s011
                                 + 0.754068 w_k1_KO_s011
                                 + 0.592253 w_k1_MAT_s011
                                 + 0.822326 w_k1_MSFT_s011
                                 + 0.827042 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.0105 w_k1_ADM_s011
                                 + 1.22786 w_k1_AXP_s011
                                 + 1.25291 w_k1_DIS_s011 + 1.1848 w_k1_FDX_s011
                                 + 0.893031 w_k1_IBM_s011
                                 + 1.03639 w_k1_INTC_s011
                                 + 1.07332 w_k1_KO_s011 + 1.06876 w_k1_MAT_s011
                                 + 1.16713 w_k1_MSFT_s011 + 1.1041 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.16199 w_k1_ADM_s011
                                 + 1.29421 w_k1_AXP_s011
                                 + 1.33963 w_k1_DIS_s011
                                 + 1.22292 w_k1_FDX_s011
                                 + 1.49905 w_k1_IBM_s011
                                 + 1.36135 w_k1_INTC_s011
                                 + 1.33309 w_k1_KO_s011 + 1.06217 w_k1_MAT_s011
                                 + 1.18206 w_k1_MSFT_s011
                                 + 1.13977 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.05365 w_k1_ADM_s011
                                 + 0.769141 w_k1_AXP_s011
                                 + 0.913335 w_k1_DIS_s011
                                 + 0.875106 w_k1_FDX_s011
                                 + 0.907773 w_k1_IBM_s011
                                 + 0.740133 w_k1_INTC_s011
                                 + 0.839971 w_k1_KO_s011
                                 + 0.95332 w_k1_MAT_s011
                                 + 0.821158 w_k1_MSFT_s011
                                 + 0.953147 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.27341 w_k1_ADM_s011
                                 + 1.15301 w_k1_AXP_s011
                                 + 1.05692 w_k1_DIS_s011
                                 + 1.19081 w_k1_FDX_s011
                                 + 0.943647 w_k1_IBM_s011
                                 + 0.908397 w_k1_INTC_s011
                                 + 0.911338 w_k1_KO_s011 + 1.0564 w_k1_MAT_s011
                                 + 0.900937 w_k1_MSFT_s011
                                 + 1.08336 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.12433 w_k1_ADM_s011
                                 + 1.05921 w_k1_AXP_s011
                                 + 1.02894 w_k1_DIS_s011
                                 + 0.94536 w_k1_FDX_s011
                                 + 1.31665 w_k1_IBM_s011
                                 + 0.845764 w_k1_INTC_s011
                                 + 1.0271 w_k1_KO_s011 + 1.0677 w_k1_MAT_s011
                                 + 0.938143 w_k1_MSFT_s011
                                 + 1.11474 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.939783 w_k1_ADM_s011
                                 + 0.789508 w_k1_AXP_s011
                                 + 0.848039 w_k1_DIS_s011
                                 + 0.807079 w_k1_FDX_s011
                                 + 1.21268 w_k1_IBM_s011
                                 + 0.858205 w_k1_INTC_s011
                                 + 0.992906 w_k1_KO_s011
                                 + 1.07921 w_k1_MAT_s011
                                 + 0.879128 w_k1_MSFT_s011
                                 + 1.01991 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.1439 w_k1_ADM_s012
                                 + 1.16894 w_k1_AXP_s012
                                 + 1.44583 w_k1_DIS_s012
                                 + 1.31009 w_k1_FDX_s012
                                 + 0.834196 w_k1_IBM_s012
                                 + 1.01034 w_k1_INTC_s012
                                 + 1.11706 w_k1_KO_s012 + 1.08669 w_k1_MAT_s012
                                 + 1.17785 w_k1_MSFT_s012 + 1.0935 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.20159 w_k1_ADM_s012
                                 + 1.21517 w_k1_AXP_s012
                                 + 0.891162 w_k1_DIS_s012
                                 + 1.05385 w_k1_FDX_s012
                                 + 0.864505 w_k1_IBM_s012
                                 + 0.911168 w_k1_INTC_s012
                                 + 0.903251 w_k1_KO_s012
                                 + 1.00378 w_k1_MAT_s012
                                 + 0.956557 w_k1_MSFT_s012
                                 + 0.911727 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.18558 w_k1_ADM_s012
                                 + 1.64636 w_k1_AXP_s012 + 1.655 w_k1_DIS_s012
                                 + 1.38771 w_k1_FDX_s012
                                 + 1.46115 w_k1_IBM_s012
                                 + 1.2161 w_k1_INTC_s012 + 1.33331 w_k1_KO_s012
                                 + 1.87197 w_k1_MAT_s012
                                 + 1.26401 w_k1_MSFT_s012 + 1.151 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.932568 w_k1_ADM_s012
                                 + 0.920892 w_k1_AXP_s012
                                 + 1.11191 w_k1_DIS_s012
                                 + 1.07169 w_k1_FDX_s012
                                 + 0.628748 w_k1_IBM_s012
                                 + 0.990793 w_k1_INTC_s012
                                 + 0.910212 w_k1_KO_s012
                                 + 0.910305 w_k1_MAT_s012
                                 + 1.2615 w_k1_MSFT_s012
                                 + 0.970981 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.977915 w_k1_ADM_s012
                                 + 1.07485 w_k1_AXP_s012
                                 + 1.01276 w_k1_DIS_s012 + 1.031 w_k1_FDX_s012
                                 + 0.881909 w_k1_IBM_s012
                                 + 0.989307 w_k1_INTC_s012
                                 + 0.943801 w_k1_KO_s012
                                 + 1.09956 w_k1_MAT_s012
                                 + 1.08397 w_k1_MSFT_s012
                                 + 1.06325 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.942368 w_k1_ADM_s012
                                 + 0.957151 w_k1_AXP_s012
                                 + 1.02311 w_k1_DIS_s012
                                 + 0.91201 w_k1_FDX_s012
                                 + 1.08629 w_k1_IBM_s012
                                 + 0.966683 w_k1_INTC_s012
                                 + 1.03391 w_k1_KO_s012 + 1.14384 w_k1_MAT_s012
                                 + 0.864422 w_k1_MSFT_s012
                                 + 0.928666 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.764258 w_k1_ADM_s012
                                 + 0.696751 w_k1_AXP_s012
                                 + 0.673606 w_k1_DIS_s012
                                 + 0.75312 w_k1_FDX_s012
                                 + 0.806459 w_k1_IBM_s012
                                 + 0.855239 w_k1_INTC_s012
                                 + 0.799577 w_k1_KO_s012
                                 + 0.803941 w_k1_MAT_s012
                                 + 0.797381 w_k1_MSFT_s012
                                 + 0.830482 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.07907 w_k1_ADM_s012
                                 + 1.21693 w_k1_AXP_s012
                                 + 1.28195 w_k1_DIS_s012
                                 + 1.08572 w_k1_FDX_s012
                                 + 1.26653 w_k1_IBM_s012
                                 + 1.24115 w_k1_INTC_s012
                                 + 1.32308 w_k1_KO_s012 + 1.20407 w_k1_MAT_s012
                                 + 1.06751 w_k1_MSFT_s012
                                 + 1.11559 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.03914 w_k1_ADM_s013
                                 + 1.26681 w_k1_AXP_s013
                                 + 1.22928 w_k1_DIS_s013
                                 + 1.22229 w_k1_FDX_s013
                                 + 1.25828 w_k1_IBM_s013
                                 + 1.12123 w_k1_INTC_s013
                                 + 1.19073 w_k1_KO_s013 + 1.34244 w_k1_MAT_s013
                                 + 0.957099 w_k1_MSFT_s013
                                 + 1.16687 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.03014 w_k1_ADM_s013
                                 + 1.52327 w_k1_AXP_s013
                                 + 1.28415 w_k1_DIS_s013
                                 + 1.18226 w_k1_FDX_s013
                                 + 1.52158 w_k1_IBM_s013
                                 + 1.12476 w_k1_INTC_s013
                                 + 1.34215 w_k1_KO_s013 + 2.47917 w_k1_MAT_s013
                                 + 0.968358 w_k1_MSFT_s013
                                 + 1.11433 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.715785 w_k1_ADM_s013
                                 + 0.681024 w_k1_AXP_s013
                                 + 0.504286 w_k1_DIS_s013
                                 + 0.772942 w_k1_FDX_s013
                                 + 0.875091 w_k1_IBM_s013
                                 + 0.880265 w_k1_INTC_s013
                                 + 0.836748 w_k1_KO_s013
                                 + 0.839843 w_k1_MAT_s013
                                 + 0.703405 w_k1_MSFT_s013
                                 + 0.838592 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.909605 w_k1_ADM_s013
                                 + 1.08471 w_k1_AXP_s013
                                 + 1.10082 w_k1_DIS_s013
                                 + 1.06992 w_k1_FDX_s013
                                 + 1.15745 w_k1_IBM_s013
                                 + 0.974163 w_k1_INTC_s013
                                 + 1.06305 w_k1_KO_s013 + 1.47002 w_k1_MAT_s013
                                 + 0.858069 w_k1_MSFT_s013
                                 + 1.02474 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.912883 w_k1_ADM_s013
                                 + 0.939313 w_k1_AXP_s013
                                 + 0.839611 w_k1_DIS_s013
                                 + 0.954652 w_k1_FDX_s013
                                 + 0.819224 w_k1_IBM_s013
                                 + 0.857745 w_k1_INTC_s013
                                 + 0.811567 w_k1_KO_s013
                                 + 1.20192 w_k1_MAT_s013
                                 + 0.830608 w_k1_MSFT_s013
                                 + 1.10717 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.878564 w_k1_ADM_s013
                                 + 0.939078 w_k1_AXP_s013
                                 + 1.1473 w_k1_DIS_s013 + 1.03024 w_k1_FDX_s013
                                 + 0.984754 w_k1_IBM_s013
                                 + 1.16828 w_k1_INTC_s013
                                 + 0.97427 w_k1_KO_s013
                                 + 0.994216 w_k1_MAT_s013
                                 + 1.08363 w_k1_MSFT_s013
                                 + 1.01617 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.10632 w_k1_ADM_s013
                                 + 1.00463 w_k1_AXP_s013
                                 + 0.920897 w_k1_DIS_s013
                                 + 1.01456 w_k1_FDX_s013
                                 + 0.890756 w_k1_IBM_s013
                                 + 1.06066 w_k1_INTC_s013
                                 + 0.94973 w_k1_KO_s013
                                 + 0.986839 w_k1_MAT_s013
                                 + 0.840661 w_k1_MSFT_s013
                                 + 1.05115 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.868647 w_k1_ADM_s013
                                 + 0.667679 w_k1_AXP_s013
                                 + 0.900278 w_k1_DIS_s013
                                 + 0.924791 w_k1_FDX_s013
                                 + 0.678175 w_k1_IBM_s013
                                 + 1.06296 w_k1_INTC_s013
                                 + 0.788309 w_k1_KO_s013
                                 + 0.480995 w_k1_MAT_s013
                                 + 0.905563 w_k1_MSFT_s013
                                 + 0.891192 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.937456 w_k1_ADM_s014
                                 + 0.946584 w_k1_AXP_s014
                                 + 0.956099 w_k1_DIS_s014
                                 + 0.972668 w_k1_FDX_s014
                                 + 0.732857 w_k1_IBM_s014
                                 + 0.886028 w_k1_INTC_s014
                                 + 0.977236 w_k1_KO_s014
                                 + 0.694145 w_k1_MAT_s014
                                 + 0.958029 w_k1_MSFT_s014
                                 + 0.994649 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.851151 w_k1_ADM_s014
                                 + 0.66694 w_k1_AXP_s014
                                 + 0.675779 w_k1_DIS_s014
                                 + 0.695045 w_k1_FDX_s014
                                 + 0.902079 w_k1_IBM_s014
                                 + 0.701567 w_k1_INTC_s014
                                 + 0.892045 w_k1_KO_s014
                                 + 0.669573 w_k1_MAT_s014
                                 + 0.757262 w_k1_MSFT_s014
                                 + 0.803149 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.24819 w_k1_ADM_s014
                                 + 1.67861 w_k1_AXP_s014
                                 + 1.46627 w_k1_DIS_s014
                                 + 1.33861 w_k1_FDX_s014
                                 + 1.31227 w_k1_IBM_s014
                                 + 1.31216 w_k1_INTC_s014 + 1.3415 w_k1_KO_s014
                                 + 1.7738 w_k1_MAT_s014
                                 + 1.38701 w_k1_MSFT_s014
                                 + 1.32676 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.985268 w_k1_ADM_s014
                                 + 0.996725 w_k1_AXP_s014
                                 + 1.28231 w_k1_DIS_s014
                                 + 1.05221 w_k1_FDX_s014
                                 + 1.11272 w_k1_IBM_s014
                                 + 1.06398 w_k1_INTC_s014
                                 + 1.40751 w_k1_KO_s014 + 1.23633 w_k1_MAT_s014
                                 + 1.14111 w_k1_MSFT_s014
                                 + 1.07703 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.918771 w_k1_ADM_s014
                                 + 1.10829 w_k1_AXP_s014
                                 + 1.18972 w_k1_DIS_s014 + 1.0199 w_k1_FDX_s014
                                 + 1.47886 w_k1_IBM_s014
                                 + 1.15541 w_k1_INTC_s014
                                 + 1.42577 w_k1_KO_s014 + 1.13405 w_k1_MAT_s014
                                 + 1.04295 w_k1_MSFT_s014
                                 + 1.22979 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.885603 w_k1_ADM_s014
                                 + 0.841341 w_k1_AXP_s014
                                 + 0.895633 w_k1_DIS_s014
                                 + 0.781666 w_k1_FDX_s014
                                 + 1.06444 w_k1_IBM_s014
                                 + 0.84149 w_k1_INTC_s014
                                 + 1.16882 w_k1_KO_s014
                                 + 0.922899 w_k1_MAT_s014
                                 + 0.972392 w_k1_MSFT_s014
                                 + 0.941685 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.788224 w_k1_ADM_s014
                                 + 0.703449 w_k1_AXP_s014
                                 + 0.68665 w_k1_DIS_s014
                                 + 0.743745 w_k1_FDX_s014
                                 + 0.686245 w_k1_IBM_s014
                                 + 0.862475 w_k1_INTC_s014
                                 + 0.814471 w_k1_KO_s014
                                 + 0.62193 w_k1_MAT_s014
                                 + 0.828018 w_k1_MSFT_s014
                                 + 0.762564 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.09969 w_k1_ADM_s014
                                 + 1.31808 w_k1_AXP_s014
                                 + 1.50206 w_k1_DIS_s014
                                 + 1.29884 w_k1_FDX_s014
                                 + 0.880251 w_k1_IBM_s014
                                 + 1.30328 w_k1_INTC_s014 + 1.2112 w_k1_KO_s014
                                 + 0.763652 w_k1_MAT_s014
                                 + 1.19154 w_k1_MSFT_s014
                                 + 1.24387 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.02846 w_k1_ADM_s015
                                 + 0.844764 w_k1_AXP_s015
                                 + 0.737549 w_k1_DIS_s015
                                 + 0.836605 w_k1_FDX_s015
                                 + 0.873515 w_k1_IBM_s015
                                 + 0.703697 w_k1_INTC_s015
                                 + 0.896269 w_k1_KO_s015
                                 + 0.898257 w_k1_MAT_s015
                                 + 0.866963 w_k1_MSFT_s015
                                 + 0.919775 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.51191 w_k1_ADM_s015
                                 + 1.98888 w_k1_AXP_s015
                                 + 1.49161 w_k1_DIS_s015
                                 + 1.25948 w_k1_FDX_s015
                                 + 1.85354 w_k1_IBM_s015
                                 + 0.965734 w_k1_INTC_s015
                                 + 1.33224 w_k1_KO_s015 + 2.04762 w_k1_MAT_s015
                                 + 1.08301 w_k1_MSFT_s015
                                 + 1.26149 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.825097 w_k1_ADM_s015
                                 + 0.929015 w_k1_AXP_s015
                                 + 0.69732 w_k1_DIS_s015
                                 + 0.826691 w_k1_FDX_s015
                                 + 1.2953 w_k1_IBM_s015
                                 + 0.863127 w_k1_INTC_s015
                                 + 1.00965 w_k1_KO_s015 + 1.24554 w_k1_MAT_s015
                                 + 0.699609 w_k1_MSFT_s015
                                 + 0.930952 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.1027 w_k1_ADM_s015
                                 + 1.27662 w_k1_AXP_s015
                                 + 1.15608 w_k1_DIS_s015
                                 + 1.15064 w_k1_FDX_s015
                                 + 0.881351 w_k1_IBM_s015
                                 + 0.994969 w_k1_INTC_s015
                                 + 1.04817 w_k1_KO_s015
                                 + 0.941078 w_k1_MAT_s015
                                 + 1.12416 w_k1_MSFT_s015
                                 + 1.08784 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.17796 w_k1_ADM_s015
                                 + 0.99767 w_k1_AXP_s015
                                 + 1.13674 w_k1_DIS_s015
                                 + 1.04067 w_k1_FDX_s015
                                 + 0.65076 w_k1_IBM_s015
                                 + 0.855422 w_k1_INTC_s015
                                 + 0.910387 w_k1_KO_s015
                                 + 0.628807 w_k1_MAT_s015
                                 + 1.09521 w_k1_MSFT_s015
                                 + 0.914286 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.743716 w_k1_ADM_s015
                                 + 0.742165 w_k1_AXP_s015
                                 + 0.70897 w_k1_DIS_s015
                                 + 0.826441 w_k1_FDX_s015
                                 + 0.705377 w_k1_IBM_s015
                                 + 0.662276 w_k1_INTC_s015
                                 + 0.676317 w_k1_KO_s015
                                 + 0.565849 w_k1_MAT_s015
                                 + 0.822577 w_k1_MSFT_s015
                                 + 0.823273 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.3066 w_k1_ADM_s015
                                 + 1.69784 w_k1_AXP_s015
                                 + 1.46194 w_k1_DIS_s015
                                 + 1.41409 w_k1_FDX_s015
                                 + 1.18314 w_k1_IBM_s015
                                 + 1.15337 w_k1_INTC_s015
                                 + 1.16993 w_k1_KO_s015 + 1.38537 w_k1_MAT_s015
                                 + 1.19362 w_k1_MSFT_s015
                                 + 1.27342 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.07259 w_k1_ADM_s015
                                 + 1.13601 w_k1_AXP_s015
                                 + 1.10324 w_k1_DIS_s015
                                 + 1.02265 w_k1_FDX_s015
                                 + 1.13405 w_k1_IBM_s015
                                 + 0.991354 w_k1_INTC_s015
                                 + 0.907413 w_k1_KO_s015
                                 + 0.971414 w_k1_MAT_s015
                                 + 0.919091 w_k1_MSFT_s015
                                 + 0.969204 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.79182 w_k1_ADM_s016
                                 + 1.12827 w_k1_AXP_s016
                                 + 0.947715 w_k1_DIS_s016
                                 + 0.88527 w_k1_FDX_s016
                                 + 1.46428 w_k1_IBM_s016
                                 + 0.956802 w_k1_INTC_s016
                                 + 1.05201 w_k1_KO_s016 + 1.54987 w_k1_MAT_s016
                                 + 0.932514 w_k1_MSFT_s016
                                 + 0.939285 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.835021 w_k1_ADM_s016
                                 + 0.767148 w_k1_AXP_s016
                                 + 0.76637 w_k1_DIS_s016
                                 + 0.896009 w_k1_FDX_s016
                                 + 0.677685 w_k1_IBM_s016
                                 + 0.806627 w_k1_INTC_s016
                                 + 0.703416 w_k1_KO_s016
                                 + 0.732573 w_k1_MAT_s016
                                 + 0.915988 w_k1_MSFT_s016
                                 + 0.837365 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.812847 w_k1_ADM_s016
                                 + 1.16263 w_k1_AXP_s016
                                 + 1.15476 w_k1_DIS_s016 + 1.0046 w_k1_FDX_s016
                                 + 1.05863 w_k1_IBM_s016
                                 + 0.967945 w_k1_INTC_s016
                                 + 1.03461 w_k1_KO_s016 + 1.03866 w_k1_MAT_s016
                                 + 0.953847 w_k1_MSFT_s016
                                 + 1.08184 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.16158 w_k1_ADM_s016
                                 + 1.47901 w_k1_AXP_s016 + 1.4374 w_k1_DIS_s016
                                 + 1.21505 w_k1_FDX_s016
                                 + 1.39403 w_k1_IBM_s016
                                 + 1.1493 w_k1_INTC_s016 + 1.34043 w_k1_KO_s016
                                 + 1.50638 w_k1_MAT_s016
                                 + 1.26147 w_k1_MSFT_s016
                                 + 1.17355 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.3228 w_k1_ADM_s016
                                 + 1.16519 w_k1_AXP_s016
                                 + 1.51361 w_k1_DIS_s016
                                 + 1.25729 w_k1_FDX_s016
                                 + 0.794371 w_k1_IBM_s016
                                 + 1.15304 w_k1_INTC_s016
                                 + 1.08714 w_k1_KO_s016
                                 + 0.923179 w_k1_MAT_s016
                                 + 1.28549 w_k1_MSFT_s016
                                 + 1.17745 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.762256 w_k1_ADM_s016
                                 + 0.785365 w_k1_AXP_s016
                                 + 0.786831 w_k1_DIS_s016
                                 + 0.787091 w_k1_FDX_s016
                                 + 0.995291 w_k1_IBM_s016
                                 + 0.748625 w_k1_INTC_s016
                                 + 0.863717 w_k1_KO_s016
                                 + 0.965476 w_k1_MAT_s016
                                 + 0.875076 w_k1_MSFT_s016
                                 + 0.86882 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.90749 w_k1_ADM_s016
                                 + 1.15844 w_k1_AXP_s016
                                 + 1.02627 w_k1_DIS_s016
                                 + 1.02908 w_k1_FDX_s016
                                 + 0.918633 w_k1_IBM_s016
                                 + 0.903249 w_k1_INTC_s016
                                 + 0.967184 w_k1_KO_s016 + 1.3255 w_k1_MAT_s016
                                 + 1.0593 w_k1_MSFT_s016 + 1.10227 w_k1_PG_s016
                                 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.25846 w_k1_ADM_s016
                                 + 1.21463 w_k1_AXP_s016
                                 + 1.20427 w_k1_DIS_s016 + 1.1793 w_k1_FDX_s016
                                 + 0.876674 w_k1_IBM_s016
                                 + 1.0412 w_k1_INTC_s016
                                 + 0.910638 w_k1_KO_s016
                                 + 0.794786 w_k1_MAT_s016
                                 + 1.0398 w_k1_MSFT_s016 + 1.12733 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.0859 w_k1_ADM_s017
                                 + 1.01325 w_k1_AXP_s017
                                 + 0.842646 w_k1_DIS_s017
                                 + 1.04908 w_k1_FDX_s017
                                 + 0.622731 w_k1_IBM_s017
                                 + 0.864663 w_k1_INTC_s017
                                 + 0.734025 w_k1_KO_s017
                                 + 0.563916 w_k1_MAT_s017
                                 + 0.844342 w_k1_MSFT_s017
                                 + 0.90946 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.826783 w_k1_ADM_s017
                                 + 1.05858 w_k1_AXP_s017
                                 + 0.920755 w_k1_DIS_s017
                                 + 0.882357 w_k1_FDX_s017
                                 + 1.01404 w_k1_IBM_s017
                                 + 0.901267 w_k1_INTC_s017
                                 + 0.871669 w_k1_KO_s017
                                 + 0.947761 w_k1_MAT_s017
                                 + 0.851357 w_k1_MSFT_s017
                                 + 0.865693 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.29056 w_k1_ADM_s017
                                 + 1.31451 w_k1_AXP_s017
                                 + 1.29398 w_k1_DIS_s017 + 1.104 w_k1_FDX_s017
                                 + 1.57299 w_k1_IBM_s017
                                 + 0.977821 w_k1_INTC_s017
                                 + 1.23849 w_k1_KO_s017 + 1.3648 w_k1_MAT_s017
                                 + 1.06317 w_k1_MSFT_s017
                                 + 1.02472 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.911785 w_k1_ADM_s017
                                 + 0.95374 w_k1_AXP_s017
                                 + 0.763101 w_k1_DIS_s017
                                 + 0.775859 w_k1_FDX_s017
                                 + 1.3165 w_k1_IBM_s017
                                 + 0.818456 w_k1_INTC_s017
                                 + 0.951984 w_k1_KO_s017
                                 + 1.01638 w_k1_MAT_s017
                                 + 0.782899 w_k1_MSFT_s017
                                 + 0.815299 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.0455 w_k1_ADM_s017
                                 + 1.1985 w_k1_AXP_s017 + 1.08415 w_k1_DIS_s017
                                 + 1.26017 w_k1_FDX_s017
                                 + 0.779172 w_k1_IBM_s017
                                 + 1.10364 w_k1_INTC_s017
                                 + 0.840575 w_k1_KO_s017
                                 + 0.868207 w_k1_MAT_s017
                                 + 1.15533 w_k1_MSFT_s017
                                 + 0.965231 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.03731 w_k1_ADM_s017
                                 + 0.84795 w_k1_AXP_s017
                                 + 1.23443 w_k1_DIS_s017
                                 + 1.04918 w_k1_FDX_s017
                                 + 0.72918 w_k1_IBM_s017
                                 + 0.995904 w_k1_INTC_s017
                                 + 0.933971 w_k1_KO_s017
                                 + 0.592663 w_k1_MAT_s017
                                 + 1.11382 w_k1_MSFT_s017
                                 + 0.897487 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.31609 w_k1_ADM_s017
                                 + 1.46167 w_k1_AXP_s017
                                 + 1.52284 w_k1_DIS_s017
                                 + 1.38229 w_k1_FDX_s017
                                 + 1.01637 w_k1_IBM_s017
                                 + 1.2276 w_k1_INTC_s017 + 1.1695 w_k1_KO_s017
                                 + 1.4546 w_k1_MAT_s017
                                 + 1.48613 w_k1_MSFT_s017
                                 + 1.04743 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.791274 w_k1_ADM_s017
                                 + 0.655364 w_k1_AXP_s017
                                 + 0.675188 w_k1_DIS_s017
                                 + 0.758635 w_k1_FDX_s017
                                 + 0.614169 w_k1_IBM_s017
                                 + 0.869114 w_k1_INTC_s017
                                 + 0.692411 w_k1_KO_s017
                                 + 0.617916 w_k1_MAT_s017
                                 + 0.856665 w_k1_MSFT_s017
                                 + 0.788108 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.12116 w_k1_ADM_s100
                                 + 0.810647 w_k1_AXP_s100
                                 + 0.709448 w_k1_DIS_s100
                                 + 0.953525 w_k1_FDX_s100
                                 + 0.689685 w_k1_IBM_s100
                                 + 1.07945 w_k1_INTC_s100
                                 + 0.778097 w_k1_KO_s100
                                 + 0.803254 w_k1_MAT_s100
                                 + 1.08059 w_k1_MSFT_s100
                                 + 1.01129 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.50481 w_k1_ADM_s100
                                 + 1.44453 w_k1_AXP_s100
                                 + 1.30163 w_k1_DIS_s100
                                 + 1.17496 w_k1_FDX_s100 + 1.1092 w_k1_IBM_s100
                                 + 1.32168 w_k1_INTC_s100
                                 + 1.24954 w_k1_KO_s100 + 1.38821 w_k1_MAT_s100
                                 + 1.20855 w_k1_MSFT_s100
                                 + 1.27437 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.03503 w_k1_ADM_s100
                                 + 1.2919 w_k1_AXP_s100 + 1.1615 w_k1_DIS_s100
                                 + 1.08983 w_k1_FDX_s100
                                 + 0.953172 w_k1_IBM_s100
                                 + 1.17498 w_k1_INTC_s100
                                 + 0.971654 w_k1_KO_s100 + 1.0832 w_k1_MAT_s100
                                 + 1.02597 w_k1_MSFT_s100
                                 + 1.10255 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.24785 w_k1_ADM_s100
                                 + 1.41138 w_k1_AXP_s100
                                 + 1.20648 w_k1_DIS_s100
                                 + 1.01181 w_k1_FDX_s100
                                 + 1.51662 w_k1_IBM_s100 + 1.1 w_k1_INTC_s100
                                 + 1.10669 w_k1_KO_s100 + 1.72831 w_k1_MAT_s100
                                 + 0.997373 w_k1_MSFT_s100
                                 + 1.15356 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.2618 w_k1_ADM_s100
                                 + 1.60747 w_k1_AXP_s100
                                 + 1.75096 w_k1_DIS_s100 + 1.3501 w_k1_FDX_s100
                                 + 1.50401 w_k1_IBM_s100
                                 + 1.47911 w_k1_INTC_s100
                                 + 1.47278 w_k1_KO_s100 + 1.77255 w_k1_MAT_s100
                                 + 1.52843 w_k1_MSFT_s100
                                 + 1.43405 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.19147 w_k1_ADM_s100
                                 + 1.09233 w_k1_AXP_s100
                                 + 0.848831 w_k1_DIS_s100
                                 + 0.885155 w_k1_FDX_s100
                                 + 0.876069 w_k1_IBM_s100
                                 + 0.818831 w_k1_INTC_s100
                                 + 0.895034 w_k1_KO_s100
                                 + 1.19988 w_k1_MAT_s100
                                 + 0.968558 w_k1_MSFT_s100
                                 + 0.996841 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.10648 w_k1_ADM_s100
                                 + 1.07873 w_k1_AXP_s100
                                 + 0.80138 w_k1_DIS_s100
                                 + 0.84858 w_k1_FDX_s100
                                 + 1.23181 w_k1_IBM_s100
                                 + 0.856068 w_k1_INTC_s100
                                 + 0.869064 w_k1_KO_s100
                                 + 1.29555 w_k1_MAT_s100
                                 + 0.881517 w_k1_MSFT_s100
                                 + 1.13409 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.88446 w_k1_ADM_s100
                                 + 0.846703 w_k1_AXP_s100
                                 + 0.799237 w_k1_DIS_s100
                                 + 0.832635 w_k1_FDX_s100
                                 + 0.970455 w_k1_IBM_s100
                                 + 0.990285 w_k1_INTC_s100
                                 + 0.922585 w_k1_KO_s100
                                 + 0.685372 w_k1_MAT_s100
                                 + 0.945459 w_k1_MSFT_s100
                                 + 1.06289 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.15076 w_k1_ADM_s101
                                 + 1.23718 w_k1_AXP_s101
                                 + 1.24104 w_k1_DIS_s101
                                 + 0.985875 w_k1_FDX_s101
                                 + 1.43272 w_k1_IBM_s101
                                 + 0.970266 w_k1_INTC_s101
                                 + 1.3444 w_k1_KO_s101 + 1.2015 w_k1_MAT_s101
                                 + 0.91591 w_k1_MSFT_s101
                                 + 1.23703 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.1164 w_k1_ADM_s101
                                 + 1.18863 w_k1_AXP_s101
                                 + 1.48821 w_k1_DIS_s101
                                 + 1.27907 w_k1_FDX_s101 + 1.0652 w_k1_IBM_s101
                                 + 1.36229 w_k1_INTC_s101
                                 + 1.24523 w_k1_KO_s101
                                 + 0.784976 w_k1_MAT_s101
                                 + 1.2122 w_k1_MSFT_s101 + 1.12953 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.747396 w_k1_ADM_s101
                                 + 0.755945 w_k1_AXP_s101
                                 + 0.817126 w_k1_DIS_s101
                                 + 0.869282 w_k1_FDX_s101
                                 + 0.664819 w_k1_IBM_s101
                                 + 1.03415 w_k1_INTC_s101
                                 + 0.730854 w_k1_KO_s101
                                 + 0.360042 w_k1_MAT_s101
                                 + 0.809464 w_k1_MSFT_s101
                                 + 0.810811 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.645941 w_k1_ADM_s101
                                 + 0.638052 w_k1_AXP_s101
                                 + 0.756665 w_k1_DIS_s101
                                 + 0.781202 w_k1_FDX_s101
                                 + 0.808562 w_k1_IBM_s101
                                 + 0.891023 w_k1_INTC_s101
                                 + 0.830489 w_k1_KO_s101
                                 + 0.525229 w_k1_MAT_s101
                                 + 0.849208 w_k1_MSFT_s101
                                 + 0.845064 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.986064 w_k1_ADM_s101
                                 + 0.798341 w_k1_AXP_s101
                                 + 0.828517 w_k1_DIS_s101
                                 + 0.775133 w_k1_FDX_s101
                                 + 0.885663 w_k1_IBM_s101
                                 + 0.824171 w_k1_INTC_s101
                                 + 0.933725 w_k1_KO_s101
                                 + 0.629577 w_k1_MAT_s101
                                 + 0.796162 w_k1_MSFT_s101
                                 + 0.868116 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.16803 w_k1_ADM_s101
                                 + 0.919747 w_k1_AXP_s101
                                 + 1.20451 w_k1_DIS_s101 + 1.0202 w_k1_FDX_s101
                                 + 1.01198 w_k1_IBM_s101
                                 + 1.32897 w_k1_INTC_s101 + 1.1182 w_k1_KO_s101
                                 + 0.682495 w_k1_MAT_s101
                                 + 1.11185 w_k1_MSFT_s101
                                 + 1.08772 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.718474 w_k1_ADM_s101
                                 + 0.84344 w_k1_AXP_s101
                                 + 0.889804 w_k1_DIS_s101
                                 + 0.893509 w_k1_FDX_s101
                                 + 1.21949 w_k1_IBM_s101
                                 + 1.03348 w_k1_INTC_s101
                                 + 0.996561 w_k1_KO_s101
                                 + 0.921026 w_k1_MAT_s101
                                 + 0.916742 w_k1_MSFT_s101
                                 + 0.922171 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.09162 w_k1_ADM_s101
                                 + 1.18982 w_k1_AXP_s101
                                 + 0.976445 w_k1_DIS_s101
                                 + 0.9921 w_k1_FDX_s101
                                 + 0.939425 w_k1_IBM_s101
                                 + 1.05619 w_k1_INTC_s101
                                 + 1.03555 w_k1_KO_s101
                                 + 0.802882 w_k1_MAT_s101
                                 + 0.944807 w_k1_MSFT_s101
                                 + 0.996722 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.07077 w_k1_ADM_s102
                                 + 1.08605 w_k1_AXP_s102
                                 + 1.75389 w_k1_DIS_s102
                                 + 1.15381 w_k1_FDX_s102
                                 + 1.18559 w_k1_IBM_s102
                                 + 1.17778 w_k1_INTC_s102
                                 + 1.34365 w_k1_KO_s102
                                 + 0.919199 w_k1_MAT_s102
                                 + 1.16201 w_k1_MSFT_s102
                                 + 1.19386 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.27248 w_k1_ADM_s102
                                 + 1.79476 w_k1_AXP_s102
                                 + 1.61902 w_k1_DIS_s102
                                 + 1.48113 w_k1_FDX_s102
                                 + 1.24732 w_k1_IBM_s102
                                 + 1.14935 w_k1_INTC_s102
                                 + 1.25487 w_k1_KO_s102
                                 + 0.993911 w_k1_MAT_s102
                                 + 1.21502 w_k1_MSFT_s102
                                 + 1.05619 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.0679 w_k1_ADM_s102
                                 + 0.864654 w_k1_AXP_s102
                                 + 0.918796 w_k1_DIS_s102
                                 + 0.891626 w_k1_FDX_s102
                                 + 1.2159 w_k1_IBM_s102
                                 + 1.00639 w_k1_INTC_s102
                                 + 1.09492 w_k1_KO_s102
                                 + 0.907403 w_k1_MAT_s102
                                 + 0.967595 w_k1_MSFT_s102
                                 + 0.927214 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.964037 w_k1_ADM_s102
                                 + 1.08117 w_k1_AXP_s102 + 1.0809 w_k1_DIS_s102
                                 + 1.02399 w_k1_FDX_s102
                                 + 0.751086 w_k1_IBM_s102
                                 + 1.0171 w_k1_INTC_s102 + 0.80767 w_k1_KO_s102
                                 + 0.502976 w_k1_MAT_s102
                                 + 0.993517 w_k1_MSFT_s102
                                 + 0.884385 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.959585 w_k1_ADM_s102
                                 + 1.0521 w_k1_AXP_s102
                                 + 0.866633 w_k1_DIS_s102
                                 + 0.938545 w_k1_FDX_s102
                                 + 0.948875 w_k1_IBM_s102
                                 + 0.844118 w_k1_INTC_s102
                                 + 0.963487 w_k1_KO_s102
                                 + 1.00562 w_k1_MAT_s102
                                 + 0.905158 w_k1_MSFT_s102
                                 + 0.869767 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.832847 w_k1_ADM_s102
                                 + 0.637496 w_k1_AXP_s102
                                 + 0.615746 w_k1_DIS_s102
                                 + 0.661837 w_k1_FDX_s102
                                 + 1.00374 w_k1_IBM_s102
                                 + 0.646658 w_k1_INTC_s102
                                 + 0.875941 w_k1_KO_s102
                                 + 0.853534 w_k1_MAT_s102
                                 + 0.744631 w_k1_MSFT_s102
                                 + 0.902364 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.944153 w_k1_ADM_s102
                                 + 1.02056 w_k1_AXP_s102
                                 + 0.842982 w_k1_DIS_s102
                                 + 0.900267 w_k1_FDX_s102
                                 + 0.693155 w_k1_IBM_s102
                                 + 0.970753 w_k1_INTC_s102
                                 + 0.805718 w_k1_KO_s102
                                 + 0.582936 w_k1_MAT_s102
                                 + 0.965598 w_k1_MSFT_s102
                                 + 0.814302 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.27591 w_k1_ADM_s102
                                 + 1.42936 w_k1_AXP_s102
                                 + 1.25491 w_k1_DIS_s102
                                 + 1.19122 w_k1_FDX_s102
                                 + 1.33344 w_k1_IBM_s102
                                 + 0.929732 w_k1_INTC_s102
                                 + 1.14123 w_k1_KO_s102 + 1.18291 w_k1_MAT_s102
                                 + 0.970627 w_k1_MSFT_s102
                                 + 0.951626 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.0029 w_k1_ADM_s103
                                 + 1.30373 w_k1_AXP_s103
                                 + 1.05684 w_k1_DIS_s103
                                 + 0.988772 w_k1_FDX_s103
                                 + 1.35828 w_k1_IBM_s103
                                 + 1.02035 w_k1_INTC_s103
                                 + 1.19027 w_k1_KO_s103 + 1.55869 w_k1_MAT_s103
                                 + 1.06354 w_k1_MSFT_s103
                                 + 1.14041 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.15522 w_k1_ADM_s103
                                 + 0.78011 w_k1_AXP_s103
                                 + 1.08711 w_k1_DIS_s103
                                 + 1.03494 w_k1_FDX_s103
                                 + 0.782067 w_k1_IBM_s103
                                 + 0.909276 w_k1_INTC_s103
                                 + 0.959615 w_k1_KO_s103
                                 + 0.606558 w_k1_MAT_s103
                                 + 1.06519 w_k1_MSFT_s103 + 1.0352 w_k1_PG_s103
                                 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.59954 w_k1_ADM_s103
                                 + 0.838022 w_k1_AXP_s103
                                 + 0.751949 w_k1_DIS_s103
                                 + 0.711187 w_k1_FDX_s103
                                 + 1.01711 w_k1_IBM_s103
                                 + 0.763484 w_k1_INTC_s103
                                 + 0.860655 w_k1_KO_s103
                                 + 1.14589 w_k1_MAT_s103
                                 + 0.708693 w_k1_MSFT_s103
                                 + 0.829581 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.953568 w_k1_ADM_s103
                                 + 1.54234 w_k1_AXP_s103
                                 + 1.05285 w_k1_DIS_s103
                                 + 1.14985 w_k1_FDX_s103
                                 + 0.839688 w_k1_IBM_s103
                                 + 1.07619 w_k1_INTC_s103
                                 + 0.924726 w_k1_KO_s103
                                 + 1.41048 w_k1_MAT_s103
                                 + 0.972706 w_k1_MSFT_s103
                                 + 1.02957 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.49754 w_k1_ADM_s103
                                 + 1.65598 w_k1_AXP_s103
                                 + 1.78821 w_k1_DIS_s103
                                 + 1.24516 w_k1_FDX_s103
                                 + 1.44399 w_k1_IBM_s103
                                 + 0.88217 w_k1_INTC_s103
                                 + 1.54471 w_k1_KO_s103 + 1.6714 w_k1_MAT_s103
                                 + 1.21073 w_k1_MSFT_s103 + 1.3047 w_k1_PG_s103
                                 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.47929 w_k1_ADM_s103
                                 + 1.58374 w_k1_AXP_s103
                                 + 1.45074 w_k1_DIS_s103
                                 + 1.42646 w_k1_FDX_s103
                                 + 0.969978 w_k1_IBM_s103
                                 + 1.12411 w_k1_INTC_s103
                                 + 1.05915 w_k1_KO_s103 + 1.26499 w_k1_MAT_s103
                                 + 1.20396 w_k1_MSFT_s103 + 1.286 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.08596 w_k1_ADM_s103
                                 + 1.1071 w_k1_AXP_s103 + 1.52937 w_k1_DIS_s103
                                 + 1.13564 w_k1_FDX_s103
                                 + 1.04488 w_k1_IBM_s103
                                 + 1.09652 w_k1_INTC_s103
                                 + 1.12904 w_k1_KO_s103 + 1.15813 w_k1_MAT_s103
                                 + 1.23304 w_k1_MSFT_s103
                                 + 1.24153 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.932858 w_k1_ADM_s103
                                 + 1.23044 w_k1_AXP_s103
                                 + 0.814931 w_k1_DIS_s103
                                 + 0.930781 w_k1_FDX_s103
                                 + 0.969399 w_k1_IBM_s103
                                 + 1.03102 w_k1_INTC_s103
                                 + 0.937716 w_k1_KO_s103
                                 + 1.33041 w_k1_MAT_s103
                                 + 0.91939 w_k1_MSFT_s103
                                 + 0.984436 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.25747 w_k1_ADM_s104
                                 + 1.28947 w_k1_AXP_s104
                                 + 1.52159 w_k1_DIS_s104 + 1.3862 w_k1_FDX_s104
                                 + 0.703318 w_k1_IBM_s104
                                 + 1.11546 w_k1_INTC_s104
                                 + 1.19398 w_k1_KO_s104 + 1.13117 w_k1_MAT_s104
                                 + 1.41473 w_k1_MSFT_s104 + 1.1977 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.13677 w_k1_ADM_s104
                                 + 0.77746 w_k1_AXP_s104
                                 + 0.794551 w_k1_DIS_s104
                                 + 0.93281 w_k1_FDX_s104
                                 + 0.521028 w_k1_IBM_s104
                                 + 0.766123 w_k1_INTC_s104
                                 + 0.724789 w_k1_KO_s104 + 0.6751 w_k1_MAT_s104
                                 + 0.933424 w_k1_MSFT_s104
                                 + 0.825586 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.02997 w_k1_ADM_s104
                                 + 1.10886 w_k1_AXP_s104
                                 + 1.25211 w_k1_DIS_s104
                                 + 1.12218 w_k1_FDX_s104
                                 + 1.23013 w_k1_IBM_s104
                                 + 1.30359 w_k1_INTC_s104
                                 + 1.19369 w_k1_KO_s104 + 1.22269 w_k1_MAT_s104
                                 + 1.11349 w_k1_MSFT_s104 + 1.1374 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.871098 w_k1_ADM_s104
                                 + 0.856075 w_k1_AXP_s104
                                 + 0.96448 w_k1_DIS_s104
                                 + 0.979904 w_k1_FDX_s104
                                 + 0.922735 w_k1_IBM_s104
                                 + 0.978936 w_k1_INTC_s104
                                 + 1.0927 w_k1_KO_s104 + 1.05405 w_k1_MAT_s104
                                 + 1.13846 w_k1_MSFT_s104
                                 + 1.09293 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.579629 w_k1_ADM_s104
                                 + 0.607716 w_k1_AXP_s104
                                 + 0.616779 w_k1_DIS_s104
                                 + 0.664229 w_k1_FDX_s104
                                 + 1.13739 w_k1_IBM_s104
                                 + 0.86343 w_k1_INTC_s104
                                 + 0.892629 w_k1_KO_s104
                                 + 1.09456 w_k1_MAT_s104
                                 + 0.793029 w_k1_MSFT_s104
                                 + 0.762163 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.740416 w_k1_ADM_s104
                                 + 0.695793 w_k1_AXP_s104
                                 + 0.554099 w_k1_DIS_s104
                                 + 0.738313 w_k1_FDX_s104
                                 + 0.827077 w_k1_IBM_s104
                                 + 0.71388 w_k1_INTC_s104
                                 + 0.657362 w_k1_KO_s104
                                 + 0.845641 w_k1_MAT_s104
                                 + 0.780149 w_k1_MSFT_s104
                                 + 0.807351 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.884861 w_k1_ADM_s104
                                 + 0.988349 w_k1_AXP_s104
                                 + 0.697803 w_k1_DIS_s104
                                 + 0.874195 w_k1_FDX_s104
                                 + 1.43966 w_k1_IBM_s104
                                 + 1.03936 w_k1_INTC_s104
                                 + 1.13456 w_k1_KO_s104 + 1.84461 w_k1_MAT_s104
                                 + 0.979063 w_k1_MSFT_s104
                                 + 1.10724 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.30934 w_k1_ADM_s104
                                 + 1.42534 w_k1_AXP_s104
                                 + 1.38165 w_k1_DIS_s104
                                 + 1.46437 w_k1_FDX_s104 + 1.0979 w_k1_IBM_s104
                                 + 1.51349 w_k1_INTC_s104
                                 + 1.45944 w_k1_KO_s104 + 1.22834 w_k1_MAT_s104
                                 + 1.42946 w_k1_MSFT_s104
                                 + 1.33314 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.947797 w_k1_ADM_s105
                                 + 1.08949 w_k1_AXP_s105
                                 + 1.06676 w_k1_DIS_s105
                                 + 1.01644 w_k1_FDX_s105
                                 + 0.961766 w_k1_IBM_s105
                                 + 1.15221 w_k1_INTC_s105
                                 + 1.00705 w_k1_KO_s105 + 1.24434 w_k1_MAT_s105
                                 + 1.11236 w_k1_MSFT_s105
                                 + 1.09179 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.26769 w_k1_ADM_s105
                                 + 1.74078 w_k1_AXP_s105 + 1.5245 w_k1_DIS_s105
                                 + 1.43241 w_k1_FDX_s105
                                 + 1.16915 w_k1_IBM_s105
                                 + 1.08779 w_k1_INTC_s105
                                 + 1.10618 w_k1_KO_s105 + 1.50569 w_k1_MAT_s105
                                 + 1.21349 w_k1_MSFT_s105
                                 + 1.33783 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.89968 w_k1_ADM_s105
                                 + 0.985716 w_k1_AXP_s105
                                 + 0.921118 w_k1_DIS_s105
                                 + 1.0099 w_k1_FDX_s105 + 0.72147 w_k1_IBM_s105
                                 + 0.81715 w_k1_INTC_s105
                                 + 0.690001 w_k1_KO_s105
                                 + 0.942961 w_k1_MAT_s105
                                 + 0.910088 w_k1_MSFT_s105
                                 + 1.06857 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.30342 w_k1_ADM_s105
                                 + 1.1708 w_k1_AXP_s105 + 1.35241 w_k1_DIS_s105
                                 + 1.2491 w_k1_FDX_s105
                                 + 0.860199 w_k1_IBM_s105
                                 + 0.975649 w_k1_INTC_s105
                                 + 0.826773 w_k1_KO_s105
                                 + 0.914386 w_k1_MAT_s105
                                 + 1.09602 w_k1_MSFT_s105
                                 + 1.08571 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.637318 w_k1_ADM_s105
                                 + 0.726349 w_k1_AXP_s105
                                 + 0.73774 w_k1_DIS_s105
                                 + 0.788035 w_k1_FDX_s105
                                 + 0.918804 w_k1_IBM_s105
                                 + 0.853988 w_k1_INTC_s105
                                 + 0.776098 w_k1_KO_s105
                                 + 1.00759 w_k1_MAT_s105
                                 + 0.970988 w_k1_MSFT_s105
                                 + 1.04491 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.30935 w_k1_ADM_s105
                                 + 1.11431 w_k1_AXP_s105
                                 + 1.30699 w_k1_DIS_s105
                                 + 1.07987 w_k1_FDX_s105
                                 + 1.74727 w_k1_IBM_s105
                                 + 0.946965 w_k1_INTC_s105
                                 + 1.43346 w_k1_KO_s105 + 1.82737 w_k1_MAT_s105
                                 + 1.08363 w_k1_MSFT_s105
                                 + 1.42284 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.776428 w_k1_ADM_s105
                                 + 0.694796 w_k1_AXP_s105
                                 + 0.837316 w_k1_DIS_s105
                                 + 0.913318 w_k1_FDX_s105
                                 + 0.564065 w_k1_IBM_s105
                                 + 0.806343 w_k1_INTC_s105
                                 + 0.639359 w_k1_KO_s105
                                 + 0.555531 w_k1_MAT_s105
                                 + 0.840125 w_k1_MSFT_s105
                                 + 0.965785 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.747915 w_k1_ADM_s105
                                 + 0.907523 w_k1_AXP_s105
                                 + 1.04246 w_k1_DIS_s105
                                 + 0.895533 w_k1_FDX_s105
                                 + 1.24211 w_k1_IBM_s105
                                 + 0.994445 w_k1_INTC_s105
                                 + 0.922978 w_k1_KO_s105
                                 + 1.25365 w_k1_MAT_s105
                                 + 0.91578 w_k1_MSFT_s105
                                 + 1.09996 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.837516 w_k1_ADM_s106
                                 + 1.53394 w_k1_AXP_s106
                                 + 1.71187 w_k1_DIS_s106
                                 + 1.38206 w_k1_FDX_s106
                                 + 1.03337 w_k1_IBM_s106
                                 + 1.26137 w_k1_INTC_s106
                                 + 1.13809 w_k1_KO_s106 + 1.28616 w_k1_MAT_s106
                                 + 1.27983 w_k1_MSFT_s106
                                 + 1.10305 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.87849 w_k1_ADM_s106
                                 + 1.17627 w_k1_AXP_s106
                                 + 1.24255 w_k1_DIS_s106
                                 + 1.05383 w_k1_FDX_s106
                                 + 1.58106 w_k1_IBM_s106
                                 + 1.0304 w_k1_INTC_s106 + 1.12513 w_k1_KO_s106
                                 + 1.4264 w_k1_MAT_s106
                                 + 0.987456 w_k1_MSFT_s106
                                 + 1.07262 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.08005 w_k1_ADM_s106
                                 + 1.6309 w_k1_AXP_s106 + 1.35779 w_k1_DIS_s106
                                 + 1.1594 w_k1_FDX_s106 + 1.39227 w_k1_IBM_s106
                                 + 1.30325 w_k1_INTC_s106
                                 + 1.30541 w_k1_KO_s106 + 1.8354 w_k1_MAT_s106
                                 + 1.15722 w_k1_MSFT_s106
                                 + 1.08836 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.652451 w_k1_ADM_s106
                                 + 0.808105 w_k1_AXP_s106
                                 + 0.962964 w_k1_DIS_s106
                                 + 0.83484 w_k1_FDX_s106
                                 + 1.14608 w_k1_IBM_s106
                                 + 1.02916 w_k1_INTC_s106
                                 + 0.995713 w_k1_KO_s106
                                 + 1.02195 w_k1_MAT_s106
                                 + 0.918099 w_k1_MSFT_s106
                                 + 0.888104 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.663413 w_k1_ADM_s106
                                 + 0.703759 w_k1_AXP_s106
                                 + 0.583709 w_k1_DIS_s106
                                 + 0.705277 w_k1_FDX_s106
                                 + 0.950638 w_k1_IBM_s106
                                 + 0.651162 w_k1_INTC_s106
                                 + 0.688394 w_k1_KO_s106
                                 + 0.742246 w_k1_MAT_s106
                                 + 0.729015 w_k1_MSFT_s106
                                 + 0.818274 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.10229 w_k1_ADM_s106
                                 + 1.09523 w_k1_AXP_s106
                                 + 1.06076 w_k1_DIS_s106
                                 + 0.985427 w_k1_FDX_s106
                                 + 1.18872 w_k1_IBM_s106
                                 + 0.982933 w_k1_INTC_s106
                                 + 1.2341 w_k1_KO_s106 + 1.15857 w_k1_MAT_s106
                                 + 1.05997 w_k1_MSFT_s106
                                 + 1.11104 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.844526 w_k1_ADM_s106
                                 + 1.08766 w_k1_AXP_s106
                                 + 0.972454 w_k1_DIS_s106
                                 + 0.983396 w_k1_FDX_s106
                                 + 0.885117 w_k1_IBM_s106
                                 + 1.04474 w_k1_INTC_s106
                                 + 0.843117 w_k1_KO_s106
                                 + 0.88547 w_k1_MAT_s106
                                 + 1.02259 w_k1_MSFT_s106
                                 + 0.957722 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.706069 w_k1_ADM_s106
                                 + 0.972345 w_k1_AXP_s106
                                 + 0.877983 w_k1_DIS_s106
                                 + 0.916522 w_k1_FDX_s106
                                 + 0.706182 w_k1_IBM_s106
                                 + 0.954375 w_k1_INTC_s106
                                 + 0.698873 w_k1_KO_s106
                                 + 0.754093 w_k1_MAT_s106
                                 + 0.895178 w_k1_MSFT_s106
                                 + 0.857632 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.721531 w_k1_ADM_s107
                                 + 0.708613 w_k1_AXP_s107
                                 + 0.690574 w_k1_DIS_s107
                                 + 0.7374 w_k1_FDX_s107
                                 + 0.826004 w_k1_IBM_s107
                                 + 0.86254 w_k1_INTC_s107
                                 + 0.812845 w_k1_KO_s107
                                 + 0.890236 w_k1_MAT_s107
                                 + 0.883082 w_k1_MSFT_s107
                                 + 0.893816 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.942446 w_k1_ADM_s107
                                 + 0.967009 w_k1_AXP_s107
                                 + 0.991815 w_k1_DIS_s107
                                 + 0.904698 w_k1_FDX_s107
                                 + 0.972128 w_k1_IBM_s107
                                 + 1.04323 w_k1_INTC_s107
                                 + 0.96916 w_k1_KO_s107 + 1.08219 w_k1_MAT_s107
                                 + 0.933217 w_k1_MSFT_s107
                                 + 0.945733 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.15144 w_k1_ADM_s107
                                 + 0.964897 w_k1_AXP_s107
                                 + 1.11158 w_k1_DIS_s107
                                 + 1.11028 w_k1_FDX_s107
                                 + 0.651252 w_k1_IBM_s107
                                 + 1.0296 w_k1_INTC_s107
                                 + 0.963822 w_k1_KO_s107
                                 + 0.930465 w_k1_MAT_s107
                                 + 1.10338 w_k1_MSFT_s107
                                 + 0.914816 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.05337 w_k1_ADM_s107
                                 + 1.50084 w_k1_AXP_s107
                                 + 1.48889 w_k1_DIS_s107
                                 + 1.20898 w_k1_FDX_s107
                                 + 1.03028 w_k1_IBM_s107
                                 + 1.22287 w_k1_INTC_s107
                                 + 1.08629 w_k1_KO_s107 + 1.64368 w_k1_MAT_s107
                                 + 1.32362 w_k1_MSFT_s107 + 1.2033 w_k1_PG_s107
                                 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.35085 w_k1_ADM_s107
                                 + 0.98231 w_k1_AXP_s107
                                 + 1.33016 w_k1_DIS_s107
                                 + 1.17698 w_k1_FDX_s107
                                 + 0.995109 w_k1_IBM_s107
                                 + 1.0739 w_k1_INTC_s107 + 1.08047 w_k1_KO_s107
                                 + 0.935073 w_k1_MAT_s107
                                 + 1.0665 w_k1_MSFT_s107 + 1.06871 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.31512 w_k1_ADM_s107
                                 + 1.61123 w_k1_AXP_s107
                                 + 1.85918 w_k1_DIS_s107
                                 + 1.33484 w_k1_FDX_s107
                                 + 1.49668 w_k1_IBM_s107
                                 + 1.33032 w_k1_INTC_s107 + 1.4354 w_k1_KO_s107
                                 + 1.85533 w_k1_MAT_s107
                                 + 1.43496 w_k1_MSFT_s107
                                 + 1.27981 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.915348 w_k1_ADM_s107
                                 + 1.07157 w_k1_AXP_s107
                                 + 0.893817 w_k1_DIS_s107
                                 + 0.853977 w_k1_FDX_s107
                                 + 1.27542 w_k1_IBM_s107
                                 + 0.941851 w_k1_INTC_s107
                                 + 0.975635 w_k1_KO_s107 + 1.0257 w_k1_MAT_s107
                                 + 0.851448 w_k1_MSFT_s107
                                 + 0.987028 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.893277 w_k1_ADM_s107
                                 + 1.07422 w_k1_AXP_s107
                                 + 1.09939 w_k1_DIS_s107
                                 + 0.942391 w_k1_FDX_s107
                                 + 1.41218 w_k1_IBM_s107
                                 + 0.992569 w_k1_INTC_s107
                                 + 1.06114 w_k1_KO_s107 + 1.74851 w_k1_MAT_s107
                                 + 0.962004 w_k1_MSFT_s107
                                 + 0.912891 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.879893 w_k1_ADM_s110
                                 + 0.860779 w_k1_AXP_s110
                                 + 0.977504 w_k1_DIS_s110
                                 + 1.00712 w_k1_FDX_s110
                                 + 1.06505 w_k1_IBM_s110
                                 + 1.03874 w_k1_INTC_s110
                                 + 1.04631 w_k1_KO_s110
                                 + 0.940522 w_k1_MAT_s110
                                 + 1.08879 w_k1_MSFT_s110
                                 + 1.08024 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.844913 w_k1_ADM_s110
                                 + 0.694867 w_k1_AXP_s110
                                 + 0.710583 w_k1_DIS_s110
                                 + 0.827793 w_k1_FDX_s110
                                 + 0.788128 w_k1_IBM_s110
                                 + 0.912292 w_k1_INTC_s110
                                 + 0.683869 w_k1_KO_s110
                                 + 0.499177 w_k1_MAT_s110
                                 + 0.846885 w_k1_MSFT_s110
                                 + 0.813828 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.20622 w_k1_ADM_s110
                                 + 0.975471 w_k1_AXP_s110
                                 + 1.27655 w_k1_DIS_s110 + 1.0617 w_k1_FDX_s110
                                 + 1.36662 w_k1_IBM_s110
                                 + 0.949289 w_k1_INTC_s110
                                 + 1.26435 w_k1_KO_s110 + 0.94627 w_k1_MAT_s110
                                 + 1.18126 w_k1_MSFT_s110
                                 + 1.21399 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.913346 w_k1_ADM_s110
                                 + 0.813615 w_k1_AXP_s110
                                 + 0.955532 w_k1_DIS_s110
                                 + 0.901285 w_k1_FDX_s110
                                 + 1.29667 w_k1_IBM_s110
                                 + 0.937891 w_k1_INTC_s110
                                 + 0.991509 w_k1_KO_s110
                                 + 1.03357 w_k1_MAT_s110
                                 + 0.960792 w_k1_MSFT_s110
                                 + 0.976186 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.17793 w_k1_ADM_s110
                                 + 1.20557 w_k1_AXP_s110
                                 + 1.34951 w_k1_DIS_s110 + 1.2692 w_k1_FDX_s110
                                 + 0.86754 w_k1_IBM_s110
                                 + 1.06978 w_k1_INTC_s110
                                 + 1.11524 w_k1_KO_s110
                                 + 0.786748 w_k1_MAT_s110
                                 + 1.21933 w_k1_MSFT_s110
                                 + 0.998123 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.99092 w_k1_ADM_s110
                                 + 1.34317 w_k1_AXP_s110
                                 + 1.68701 w_k1_DIS_s110
                                 + 1.27318 w_k1_FDX_s110
                                 + 1.68781 w_k1_IBM_s110
                                 + 1.19093 w_k1_INTC_s110
                                 + 1.45584 w_k1_KO_s110 + 1.49625 w_k1_MAT_s110
                                 + 1.17458 w_k1_MSFT_s110
                                 + 1.23967 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.660455 w_k1_ADM_s110
                                 + 0.890261 w_k1_AXP_s110
                                 + 1.27489 w_k1_DIS_s110
                                 + 0.915593 w_k1_FDX_s110
                                 + 1.17818 w_k1_IBM_s110
                                 + 1.06674 w_k1_INTC_s110
                                 + 1.07994 w_k1_KO_s110 + 0.76271 w_k1_MAT_s110
                                 + 1.02955 w_k1_MSFT_s110
                                 + 0.974804 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.948549 w_k1_ADM_s110
                                 + 1.14436 w_k1_AXP_s110
                                 + 0.913818 w_k1_DIS_s110
                                 + 1.00983 w_k1_FDX_s110
                                 + 0.969984 w_k1_IBM_s110
                                 + 1.02196 w_k1_INTC_s110
                                 + 0.925966 w_k1_KO_s110
                                 + 0.733096 w_k1_MAT_s110
                                 + 1.03072 w_k1_MSFT_s110
                                 + 0.964598 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.10146 w_k1_ADM_s111
                                 + 0.992 w_k1_AXP_s111 + 0.972902 w_k1_DIS_s111
                                 + 0.856464 w_k1_FDX_s111
                                 + 1.73518 w_k1_IBM_s111
                                 + 1.07594 w_k1_INTC_s111
                                 + 1.34199 w_k1_KO_s111 + 1.88932 w_k1_MAT_s111
                                 + 0.977208 w_k1_MSFT_s111
                                 + 1.05878 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.895584 w_k1_ADM_s111
                                 + 0.827024 w_k1_AXP_s111
                                 + 0.872696 w_k1_DIS_s111
                                 + 0.802669 w_k1_FDX_s111
                                 + 1.09167 w_k1_IBM_s111
                                 + 0.780501 w_k1_INTC_s111
                                 + 0.872951 w_k1_KO_s111
                                 + 1.23158 w_k1_MAT_s111
                                 + 0.994115 w_k1_MSFT_s111
                                 + 0.849499 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.27265 w_k1_ADM_s111
                                 + 1.35579 w_k1_AXP_s111
                                 + 1.17366 w_k1_DIS_s111
                                 + 1.03146 w_k1_FDX_s111
                                 + 1.61298 w_k1_IBM_s111
                                 + 1.25816 w_k1_INTC_s111
                                 + 1.36945 w_k1_KO_s111 + 2.09094 w_k1_MAT_s111
                                 + 1.21411 w_k1_MSFT_s111
                                 + 0.951866 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.00607 w_k1_ADM_s111
                                 + 0.87424 w_k1_AXP_s111
                                 + 0.681187 w_k1_DIS_s111
                                 + 0.918615 w_k1_FDX_s111
                                 + 0.621272 w_k1_IBM_s111
                                 + 0.830593 w_k1_INTC_s111
                                 + 0.73773 w_k1_KO_s111
                                 + 0.853375 w_k1_MAT_s111
                                 + 0.937085 w_k1_MSFT_s111
                                 + 0.790818 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.928087 w_k1_ADM_s111
                                 + 0.663161 w_k1_AXP_s111
                                 + 0.922949 w_k1_DIS_s111
                                 + 0.967647 w_k1_FDX_s111
                                 + 0.820462 w_k1_IBM_s111
                                 + 0.90198 w_k1_INTC_s111
                                 + 0.890124 w_k1_KO_s111
                                 + 1.20692 w_k1_MAT_s111
                                 + 1.06621 w_k1_MSFT_s111
                                 + 1.04235 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.498 w_k1_ADM_s111
                                 + 1.30191 w_k1_AXP_s111
                                 + 1.32938 w_k1_DIS_s111
                                 + 1.28207 w_k1_FDX_s111 + 1.1381 w_k1_IBM_s111
                                 + 1.35885 w_k1_INTC_s111
                                 + 1.24612 w_k1_KO_s111 + 1.0132 w_k1_MAT_s111
                                 + 1.33972 w_k1_MSFT_s111
                                 + 1.16285 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.80003 w_k1_ADM_s111
                                 + 0.941074 w_k1_AXP_s111
                                 + 0.566243 w_k1_DIS_s111
                                 + 0.780413 w_k1_FDX_s111
                                 + 0.85234 w_k1_IBM_s111
                                 + 0.819246 w_k1_INTC_s111
                                 + 0.839627 w_k1_KO_s111
                                 + 1.45969 w_k1_MAT_s111
                                 + 0.792062 w_k1_MSFT_s111
                                 + 0.916821 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.975742 w_k1_ADM_s111
                                 + 0.954891 w_k1_AXP_s111
                                 + 1.31714 w_k1_DIS_s111
                                 + 1.04159 w_k1_FDX_s111
                                 + 1.09771 w_k1_IBM_s111
                                 + 0.933926 w_k1_INTC_s111
                                 + 1.24062 w_k1_KO_s111 + 1.60157 w_k1_MAT_s111
                                 + 1.17073 w_k1_MSFT_s111
                                 + 1.00486 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.895023 w_k1_ADM_s112
                                 + 1.37244 w_k1_AXP_s112 + 1.6102 w_k1_DIS_s112
                                 + 1.20641 w_k1_FDX_s112
                                 + 1.39338 w_k1_IBM_s112
                                 + 1.07737 w_k1_INTC_s112 + 1.4898 w_k1_KO_s112
                                 + 1.04918 w_k1_MAT_s112
                                 + 1.09804 w_k1_MSFT_s112 + 1.0791 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.75911 w_k1_ADM_s112
                                 + 0.662533 w_k1_AXP_s112
                                 + 0.63052 w_k1_DIS_s112
                                 + 0.770479 w_k1_FDX_s112
                                 + 0.75419 w_k1_IBM_s112
                                 + 0.801363 w_k1_INTC_s112
                                 + 0.739469 w_k1_KO_s112
                                 + 0.551314 w_k1_MAT_s112
                                 + 0.729537 w_k1_MSFT_s112
                                 + 0.831453 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.99825 w_k1_ADM_s112
                                 + 1.01803 w_k1_AXP_s112
                                 + 1.18963 w_k1_DIS_s112
                                 + 1.17751 w_k1_FDX_s112
                                 + 0.862053 w_k1_IBM_s112
                                 + 0.884294 w_k1_INTC_s112
                                 + 0.92071 w_k1_KO_s112
                                 + 0.747272 w_k1_MAT_s112
                                 + 1.01275 w_k1_MSFT_s112
                                 + 1.10754 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.739772 w_k1_ADM_s112
                                 + 0.849311 w_k1_AXP_s112
                                 + 0.961587 w_k1_DIS_s112
                                 + 0.934962 w_k1_FDX_s112
                                 + 1.04109 w_k1_IBM_s112
                                 + 0.87463 w_k1_INTC_s112
                                 + 0.926498 w_k1_KO_s112
                                 + 0.996467 w_k1_MAT_s112
                                 + 0.945837 w_k1_MSFT_s112
                                 + 0.945544 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.92593 w_k1_ADM_s112
                                 + 1.13863 w_k1_AXP_s112 + 1.3583 w_k1_DIS_s112
                                 + 1.25441 w_k1_FDX_s112
                                 + 1.10315 w_k1_IBM_s112
                                 + 1.14806 w_k1_INTC_s112
                                 + 1.10368 w_k1_KO_s112 + 1.26087 w_k1_MAT_s112
                                 + 1.21551 w_k1_MSFT_s112 + 1.2035 w_k1_PG_s112
                                 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.916864 w_k1_ADM_s112
                                 + 1.30192 w_k1_AXP_s112
                                 + 1.36167 w_k1_DIS_s112
                                 + 1.15132 w_k1_FDX_s112
                                 + 1.69215 w_k1_IBM_s112
                                 + 0.975291 w_k1_INTC_s112
                                 + 1.31685 w_k1_KO_s112 + 1.70307 w_k1_MAT_s112
                                 + 1.14898 w_k1_MSFT_s112
                                 + 1.17167 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.667204 w_k1_ADM_s112
                                 + 0.693993 w_k1_AXP_s112
                                 + 0.77896 w_k1_DIS_s112
                                 + 0.780305 w_k1_FDX_s112
                                 + 1.28592 w_k1_IBM_s112
                                 + 0.805807 w_k1_INTC_s112
                                 + 1.13914 w_k1_KO_s112
                                 + 0.872617 w_k1_MAT_s112
                                 + 0.814335 w_k1_MSFT_s112
                                 + 1.0365 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.785628 w_k1_ADM_s112
                                 + 1.02627 w_k1_AXP_s112
                                 + 0.962458 w_k1_DIS_s112
                                 + 1.09725 w_k1_FDX_s112
                                 + 0.80116 w_k1_IBM_s112
                                 + 0.932131 w_k1_INTC_s112
                                 + 0.839929 w_k1_KO_s112
                                 + 0.744218 w_k1_MAT_s112
                                 + 0.969145 w_k1_MSFT_s112
                                 + 0.952671 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.979773 w_k1_ADM_s113
                                 + 1.52279 w_k1_AXP_s113
                                 + 1.72185 w_k1_DIS_s113
                                 + 1.43538 w_k1_FDX_s113
                                 + 1.33888 w_k1_IBM_s113
                                 + 1.2723 w_k1_INTC_s113 + 1.34475 w_k1_KO_s113
                                 + 1.93879 w_k1_MAT_s113
                                 + 1.1447 w_k1_MSFT_s113 + 1.34 w_k1_PG_s113
                                 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.870119 w_k1_ADM_s113
                                 + 0.89544 w_k1_AXP_s113
                                 + 1.21476 w_k1_DIS_s113
                                 + 1.13933 w_k1_FDX_s113
                                 + 0.819003 w_k1_IBM_s113
                                 + 1.06774 w_k1_INTC_s113
                                 + 1.01354 w_k1_KO_s113
                                 + 0.668338 w_k1_MAT_s113
                                 + 0.954919 w_k1_MSFT_s113
                                 + 0.965144 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.527514 w_k1_ADM_s113
                                 + 0.772328 w_k1_AXP_s113
                                 + 0.570119 w_k1_DIS_s113
                                 + 0.776251 w_k1_FDX_s113
                                 + 0.716487 w_k1_IBM_s113
                                 + 0.90071 w_k1_INTC_s113
                                 + 0.735551 w_k1_KO_s113
                                 + 0.823224 w_k1_MAT_s113
                                 + 0.708388 w_k1_MSFT_s113
                                 + 0.781849 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.811996 w_k1_ADM_s113
                                 + 0.869011 w_k1_AXP_s113
                                 + 0.804592 w_k1_DIS_s113
                                 + 0.874731 w_k1_FDX_s113
                                 + 0.825793 w_k1_IBM_s113
                                 + 0.832673 w_k1_INTC_s113
                                 + 0.79214 w_k1_KO_s113 + 0.94405 w_k1_MAT_s113
                                 + 0.763014 w_k1_MSFT_s113
                                 + 0.916885 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.644061 w_k1_ADM_s113
                                 + 0.842694 w_k1_AXP_s113
                                 + 0.659773 w_k1_DIS_s113
                                 + 0.697819 w_k1_FDX_s113
                                 + 1.03331 w_k1_IBM_s113
                                 + 0.930547 w_k1_INTC_s113
                                 + 0.899412 w_k1_KO_s113
                                 + 1.40838 w_k1_MAT_s113
                                 + 0.658076 w_k1_MSFT_s113
                                 + 0.889488 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.741554 w_k1_ADM_s113
                                 + 1.44921 w_k1_AXP_s113
                                 + 0.965441 w_k1_DIS_s113
                                 + 0.950353 w_k1_FDX_s113
                                 + 1.38918 w_k1_IBM_s113
                                 + 0.884683 w_k1_INTC_s113
                                 + 0.982478 w_k1_KO_s113
                                 + 2.00148 w_k1_MAT_s113
                                 + 0.735586 w_k1_MSFT_s113
                                 + 0.989403 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.851218 w_k1_ADM_s113
                                 + 1.25628 w_k1_AXP_s113
                                 + 1.14641 w_k1_DIS_s113
                                 + 1.21472 w_k1_FDX_s113
                                 + 0.973849 w_k1_IBM_s113
                                 + 1.20063 w_k1_INTC_s113
                                 + 1.05072 w_k1_KO_s113 + 1.58798 w_k1_MAT_s113
                                 + 1.00257 w_k1_MSFT_s113
                                 + 1.12834 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.833606 w_k1_ADM_s113
                                 + 0.871744 w_k1_AXP_s113
                                 + 1.14302 w_k1_DIS_s113
                                 + 1.06054 w_k1_FDX_s113
                                 + 1.48878 w_k1_IBM_s113
                                 + 0.958936 w_k1_INTC_s113
                                 + 1.19174 w_k1_KO_s113 + 1.49425 w_k1_MAT_s113
                                 + 0.92338 w_k1_MSFT_s113
                                 + 1.06697 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.981358 w_k1_ADM_s114
                                 + 0.739116 w_k1_AXP_s114
                                 + 0.932896 w_k1_DIS_s114
                                 + 0.946899 w_k1_FDX_s114
                                 + 0.725325 w_k1_IBM_s114
                                 + 1.07331 w_k1_INTC_s114
                                 + 0.950931 w_k1_KO_s114
                                 + 0.530252 w_k1_MAT_s114
                                 + 1.10122 w_k1_MSFT_s114
                                 + 1.03114 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.868165 w_k1_ADM_s114
                                 + 0.781332 w_k1_AXP_s114
                                 + 0.897618 w_k1_DIS_s114
                                 + 0.803838 w_k1_FDX_s114
                                 + 1.11056 w_k1_IBM_s114
                                 + 1.06419 w_k1_INTC_s114
                                 + 1.13301 w_k1_KO_s114
                                 + 0.732514 w_k1_MAT_s114
                                 + 1.00876 w_k1_MSFT_s114
                                 + 1.04447 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.05018 w_k1_ADM_s114
                                 + 1.17709 w_k1_AXP_s114
                                 + 1.45684 w_k1_DIS_s114
                                 + 1.15142 w_k1_FDX_s114
                                 + 1.28798 w_k1_IBM_s114
                                 + 1.17455 w_k1_INTC_s114
                                 + 1.31998 w_k1_KO_s114 + 1.27979 w_k1_MAT_s114
                                 + 1.16065 w_k1_MSFT_s114
                                 + 1.23705 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.818012 w_k1_ADM_s114
                                 + 0.789264 w_k1_AXP_s114
                                 + 0.639977 w_k1_DIS_s114
                                 + 0.826814 w_k1_FDX_s114
                                 + 0.772742 w_k1_IBM_s114
                                 + 0.988935 w_k1_INTC_s114
                                 + 0.742531 w_k1_KO_s114
                                 + 0.64583 w_k1_MAT_s114
                                 + 0.971602 w_k1_MSFT_s114
                                 + 0.847857 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.80634 w_k1_ADM_s114
                                 + 1.02464 w_k1_AXP_s114
                                 + 0.834579 w_k1_DIS_s114
                                 + 0.942827 w_k1_FDX_s114
                                 + 1.17989 w_k1_IBM_s114
                                 + 0.993184 w_k1_INTC_s114
                                 + 0.986261 w_k1_KO_s114
                                 + 1.20646 w_k1_MAT_s114
                                 + 0.936589 w_k1_MSFT_s114
                                 + 1.13366 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.03603 w_k1_ADM_s114
                                 + 0.904675 w_k1_AXP_s114
                                 + 1.06291 w_k1_DIS_s114
                                 + 1.13585 w_k1_FDX_s114
                                 + 0.849348 w_k1_IBM_s114
                                 + 1.06137 w_k1_INTC_s114
                                 + 1.00284 w_k1_KO_s114
                                 + 0.995976 w_k1_MAT_s114
                                 + 1.19681 w_k1_MSFT_s114
                                 + 1.04748 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.19552 w_k1_ADM_s114
                                 + 1.45729 w_k1_AXP_s114
                                 + 1.13466 w_k1_DIS_s114
                                 + 1.14333 w_k1_FDX_s114
                                 + 1.17053 w_k1_IBM_s114
                                 + 1.40494 w_k1_INTC_s114
                                 + 1.15872 w_k1_KO_s114 + 1.40399 w_k1_MAT_s114
                                 + 1.21174 w_k1_MSFT_s114
                                 + 1.17217 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.07857 w_k1_ADM_s114
                                 + 0.81217 w_k1_AXP_s114
                                 + 0.837992 w_k1_DIS_s114
                                 + 0.937361 w_k1_FDX_s114
                                 + 0.959698 w_k1_IBM_s114
                                 + 0.89599 w_k1_INTC_s114
                                 + 0.969169 w_k1_KO_s114
                                 + 0.974638 w_k1_MAT_s114
                                 + 0.942031 w_k1_MSFT_s114
                                 + 1.01896 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.24082 w_k1_ADM_s115
                                 + 1.11412 w_k1_AXP_s115
                                 + 0.833432 w_k1_DIS_s115
                                 + 0.84044 w_k1_FDX_s115
                                 + 1.04009 w_k1_IBM_s115
                                 + 0.923126 w_k1_INTC_s115
                                 + 0.95842 w_k1_KO_s115
                                 + 0.943445 w_k1_MAT_s115
                                 + 0.733824 w_k1_MSFT_s115
                                 + 0.890002 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.992654 w_k1_ADM_s115
                                 + 0.817264 w_k1_AXP_s115
                                 + 0.645534 w_k1_DIS_s115
                                 + 0.802458 w_k1_FDX_s115
                                 + 0.746137 w_k1_IBM_s115
                                 + 0.816424 w_k1_INTC_s115
                                 + 0.704199 w_k1_KO_s115
                                 + 0.671275 w_k1_MAT_s115
                                 + 0.80168 w_k1_MSFT_s115
                                 + 0.858449 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.20826 w_k1_ADM_s115
                                 + 0.997384 w_k1_AXP_s115
                                 + 1.18683 w_k1_DIS_s115
                                 + 1.01967 w_k1_FDX_s115
                                 + 0.861481 w_k1_IBM_s115
                                 + 1.2997 w_k1_INTC_s115 + 1.08841 w_k1_KO_s115
                                 + 0.720061 w_k1_MAT_s115
                                 + 1.08289 w_k1_MSFT_s115
                                 + 0.917578 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.05556 w_k1_ADM_s115
                                 + 0.603867 w_k1_AXP_s115
                                 + 0.847902 w_k1_DIS_s115
                                 + 0.845997 w_k1_FDX_s115
                                 + 0.786486 w_k1_IBM_s115
                                 + 0.855861 w_k1_INTC_s115
                                 + 0.937019 w_k1_KO_s115
                                 + 0.862687 w_k1_MAT_s115
                                 + 0.843621 w_k1_MSFT_s115
                                 + 0.765003 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.38316 w_k1_ADM_s115
                                 + 1.10669 w_k1_AXP_s115
                                 + 1.47096 w_k1_DIS_s115
                                 + 1.32525 w_k1_FDX_s115
                                 + 0.968502 w_k1_IBM_s115
                                 + 1.13958 w_k1_INTC_s115
                                 + 1.27116 w_k1_KO_s115 + 1.1123 w_k1_MAT_s115
                                 + 1.16372 w_k1_MSFT_s115
                                 + 1.14575 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.9085 w_k1_ADM_s115
                                 + 0.898524 w_k1_AXP_s115
                                 + 1.15533 w_k1_DIS_s115
                                 + 0.871276 w_k1_FDX_s115
                                 + 1.4278 w_k1_IBM_s115
                                 + 1.07716 w_k1_INTC_s115
                                 + 1.24873 w_k1_KO_s115 + 1.33824 w_k1_MAT_s115
                                 + 0.988142 w_k1_MSFT_s115
                                 + 1.10968 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.48359 w_k1_ADM_s115
                                 + 1.4426 w_k1_AXP_s115 + 1.27794 w_k1_DIS_s115
                                 + 1.23071 w_k1_FDX_s115 + 1.3279 w_k1_IBM_s115
                                 + 1.03789 w_k1_INTC_s115
                                 + 1.11745 w_k1_KO_s115 + 1.4903 w_k1_MAT_s115
                                 + 1.01304 w_k1_MSFT_s115
                                 + 1.12159 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.60325 w_k1_ADM_s115
                                 + 1.60826 w_k1_AXP_s115
                                 + 2.06418 w_k1_DIS_s115
                                 + 1.35508 w_k1_FDX_s115
                                 + 1.70358 w_k1_IBM_s115
                                 + 1.33097 w_k1_INTC_s115
                                 + 1.92935 w_k1_KO_s115 + 1.31258 w_k1_MAT_s115
                                 + 1.37661 w_k1_MSFT_s115 + 1.3134 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.831813 w_k1_ADM_s116
                                 + 0.991157 w_k1_AXP_s116
                                 + 0.799962 w_k1_DIS_s116
                                 + 0.896234 w_k1_FDX_s116
                                 + 0.938807 w_k1_IBM_s116
                                 + 0.809129 w_k1_INTC_s116
                                 + 0.816912 w_k1_KO_s116
                                 + 0.995432 w_k1_MAT_s116
                                 + 0.735277 w_k1_MSFT_s116
                                 + 0.870788 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.30745 w_k1_ADM_s116
                                 + 1.37594 w_k1_AXP_s116 + 1.1433 w_k1_DIS_s116
                                 + 1.16409 w_k1_FDX_s116
                                 + 1.46771 w_k1_IBM_s116
                                 + 1.13097 w_k1_INTC_s116
                                 + 1.11315 w_k1_KO_s116 + 1.19736 w_k1_MAT_s116
                                 + 0.91373 w_k1_MSFT_s116
                                 + 1.11123 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.813527 w_k1_ADM_s116
                                 + 0.67655 w_k1_AXP_s116
                                 + 0.613186 w_k1_DIS_s116
                                 + 0.827294 w_k1_FDX_s116
                                 + 0.716641 w_k1_IBM_s116
                                 + 0.693656 w_k1_INTC_s116
                                 + 0.621796 w_k1_KO_s116
                                 + 0.689663 w_k1_MAT_s116
                                 + 0.73016 w_k1_MSFT_s116
                                 + 0.932814 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.03319 w_k1_ADM_s116
                                 + 0.915073 w_k1_AXP_s116
                                 + 0.797616 w_k1_DIS_s116
                                 + 0.800091 w_k1_FDX_s116
                                 + 1.27837 w_k1_IBM_s116
                                 + 0.761992 w_k1_INTC_s116
                                 + 0.943262 w_k1_KO_s116
                                 + 1.05895 w_k1_MAT_s116
                                 + 0.750094 w_k1_MSFT_s116
                                 + 0.895221 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.45215 w_k1_ADM_s116
                                 + 1.16002 w_k1_AXP_s116
                                 + 0.973771 w_k1_DIS_s116
                                 + 1.2019 w_k1_FDX_s116
                                 + 0.702325 w_k1_IBM_s116
                                 + 0.797267 w_k1_INTC_s116
                                 + 0.671313 w_k1_KO_s116
                                 + 0.585765 w_k1_MAT_s116
                                 + 0.876297 w_k1_MSFT_s116
                                 + 1.00249 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.43845 w_k1_ADM_s116
                                 + 1.69778 w_k1_AXP_s116
                                 + 1.79178 w_k1_DIS_s116
                                 + 1.43411 w_k1_FDX_s116
                                 + 1.35612 w_k1_IBM_s116
                                 + 1.1405 w_k1_INTC_s116 + 1.5082 w_k1_KO_s116
                                 + 1.12432 w_k1_MAT_s116
                                 + 1.05657 w_k1_MSFT_s116
                                 + 1.26754 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.16777 w_k1_ADM_s116
                                 + 1.00877 w_k1_AXP_s116
                                 + 0.901107 w_k1_DIS_s116
                                 + 0.974666 w_k1_FDX_s116
                                 + 1.11878 w_k1_IBM_s116
                                 + 0.974376 w_k1_INTC_s116
                                 + 0.951309 w_k1_KO_s116
                                 + 0.957749 w_k1_MAT_s116
                                 + 0.9629 w_k1_MSFT_s116 + 1.09098 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.19412 w_k1_ADM_s116
                                 + 1.31731 w_k1_AXP_s116 + 1.2469 w_k1_DIS_s116
                                 + 1.19664 w_k1_FDX_s116
                                 + 1.15245 w_k1_IBM_s116
                                 + 1.15217 w_k1_INTC_s116
                                 + 1.06313 w_k1_KO_s116 + 1.12704 w_k1_MAT_s116
                                 + 1.13746 w_k1_MSFT_s116
                                 + 1.11752 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.07619 w_k1_ADM_s117
                                 + 1.05563 w_k1_AXP_s117
                                 + 1.07749 w_k1_DIS_s117
                                 + 1.04958 w_k1_FDX_s117
                                 + 0.99413 w_k1_IBM_s117
                                 + 1.25411 w_k1_INTC_s117
                                 + 0.950738 w_k1_KO_s117
                                 + 0.851908 w_k1_MAT_s117
                                 + 1.03717 w_k1_MSFT_s117
                                 + 1.00666 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.41125 w_k1_ADM_s117
                                 + 1.78989 w_k1_AXP_s117 + 1.5547 w_k1_DIS_s117
                                 + 1.29289 w_k1_FDX_s117
                                 + 1.44608 w_k1_IBM_s117
                                 + 1.37928 w_k1_INTC_s117
                                 + 1.22637 w_k1_KO_s117 + 1.77496 w_k1_MAT_s117
                                 + 1.21578 w_k1_MSFT_s117
                                 + 1.18273 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.987503 w_k1_ADM_s117
                                 + 0.827328 w_k1_AXP_s117
                                 + 0.738419 w_k1_DIS_s117
                                 + 0.833382 w_k1_FDX_s117
                                 + 0.730803 w_k1_IBM_s117
                                 + 0.870594 w_k1_INTC_s117
                                 + 0.809385 w_k1_KO_s117
                                 + 0.659897 w_k1_MAT_s117
                                 + 0.852734 w_k1_MSFT_s117
                                 + 0.763297 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.850249 w_k1_ADM_s117
                                 + 0.914317 w_k1_AXP_s117
                                 + 0.659796 w_k1_DIS_s117
                                 + 0.670763 w_k1_FDX_s117
                                 + 0.958685 w_k1_IBM_s117
                                 + 0.826312 w_k1_INTC_s117
                                 + 0.734769 w_k1_KO_s117
                                 + 0.898098 w_k1_MAT_s117
                                 + 0.662456 w_k1_MSFT_s117
                                 + 0.826184 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.13785 w_k1_ADM_s117
                                 + 1.12846 w_k1_AXP_s117
                                 + 1.00523 w_k1_DIS_s117
                                 + 0.986562 w_k1_FDX_s117
                                 + 0.859105 w_k1_IBM_s117
                                 + 1.12635 w_k1_INTC_s117
                                 + 0.97365 w_k1_KO_s117
                                 + 0.815137 w_k1_MAT_s117
                                 + 0.942418 w_k1_MSFT_s117
                                 + 0.897845 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.06354 w_k1_ADM_s117
                                 + 1.6063 w_k1_AXP_s117 + 1.12959 w_k1_DIS_s117
                                 + 1.01159 w_k1_FDX_s117
                                 + 1.92777 w_k1_IBM_s117
                                 + 1.26952 w_k1_INTC_s117
                                 + 1.16552 w_k1_KO_s117 + 1.44833 w_k1_MAT_s117
                                 + 0.829092 w_k1_MSFT_s117
                                 + 0.862622 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.932672 w_k1_ADM_s117
                                 + 1.01819 w_k1_AXP_s117
                                 + 0.986507 w_k1_DIS_s117
                                 + 0.912556 w_k1_FDX_s117
                                 + 1.2296 w_k1_IBM_s117 + 1.2005 w_k1_INTC_s117
                                 + 1.21663 w_k1_KO_s117 + 1.39294 w_k1_MAT_s117
                                 + 1.15483 w_k1_MSFT_s117
                                 + 0.970917 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.997484 w_k1_ADM_s117
                                 + 1.00541 w_k1_AXP_s117
                                 + 0.809895 w_k1_DIS_s117
                                 + 0.814419 w_k1_FDX_s117
                                 + 1.37571 w_k1_IBM_s117
                                 + 1.20336 w_k1_INTC_s117
                                 + 1.18371 w_k1_KO_s117 + 1.40488 w_k1_MAT_s117
                                 + 0.9922 w_k1_MSFT_s117 + 1.09223 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.0657222242275587 w_k1_ADM_s000
                                 + 0.0561954949288293 w_k1_ADM_s001
                                 + 0.0456055831628458 w_k1_ADM_s002
                                 + 0.0250433957169746 w_k1_ADM_s003
                                 + 0.0846897416933658 w_k1_ADM_s004
                                 + 0.03694346221097 w_k1_ADM_s005
                                 + 0.0889008333052206 w_k1_ADM_s006
                                 + 0.0685802466202496 w_k1_ADM_s007
                                 + 0.0453132323007834 w_k1_ADM_s010
                                 + 0.0729398776267012 w_k1_ADM_s011
                                 + 0.0332993032587835 w_k1_ADM_s012
                                 + 0.0382169734983981 w_k1_ADM_s013
                                 + 0.046201048496809 w_k1_ADM_s014
                                 + 0.0457288703840264 w_k1_ADM_s015
                                 + 0.0361286306599349 w_k1_ADM_s016
                                 + 0.0250697466227612 w_k1_ADM_s017
                                 + 0.00888616046380496 w_k1_ADM_s100
                                 + 0.011812671654067 w_k1_ADM_s101
                                 + 0.0079399815597356 w_k1_ADM_s102
                                 + 0.00967816478549632 w_k1_ADM_s103
                                 + 0.012880032041921 w_k1_ADM_s104
                                 + 0.0133687344167825 w_k1_ADM_s105
                                 + 0.0105144136591661 w_k1_ADM_s106
                                 + 0.00588655692386651 w_k1_ADM_s107
                                 + 0.01345272808914 w_k1_ADM_s110
                                 + 0.0219239515256465 w_k1_ADM_s111
                                 + 0.016220794775221 w_k1_ADM_s112
                                 + 0.0138493648920475 w_k1_ADM_s113
                                 + 0.0187501605071617 w_k1_ADM_s114
                                 + 0.020394785800368 w_k1_ADM_s115
                                 + 0.0143508884660698 w_k1_ADM_s116
                                 + 0.00683757410086626 w_k1_ADM_s117
                                 + 0.0601553402170514 w_k1_AXP_s000
                                 + 0.0585190807564811 w_k1_AXP_s001
                                 + 0.0506397749825757 w_k1_AXP_s002
                                 + 0.0230905614533351 w_k1_AXP_s003
                                 + 0.0849767753951475 w_k1_AXP_s004
                                 + 0.0370200011423516 w_k1_AXP_s005
                                 + 0.0772481106173439 w_k1_AXP_s006
                                 + 0.0648436834155528 w_k1_AXP_s007
                                 + 0.036914671587853 w_k1_AXP_s010
                                 + 0.0675633695830316 w_k1_AXP_s011
                                 + 0.0357342897560871 w_k1_AXP_s012
                                 + 0.042475190625557 w_k1_AXP_s013
                                 + 0.0496823388572951 w_k1_AXP_s014
                                 + 0.0499025055277448 w_k1_AXP_s015
                                 + 0.0412001811765385 w_k1_AXP_s016
                                 + 0.0262162609341704 w_k1_AXP_s017
                                 + 0.00900905504006897 w_k1_AXP_s100
                                 + 0.0118390696602213 w_k1_AXP_s101
                                 + 0.00833221713945741 w_k1_AXP_s102
                                 + 0.0110058029903496 w_k1_AXP_s103
                                 + 0.0128610632810066 w_k1_AXP_s104
                                 + 0.0141561959947932 w_k1_AXP_s105
                                 + 0.0142724408594802 w_k1_AXP_s106
                                 + 0.00591473891097378 w_k1_AXP_s107
                                 + 0.0135510739859387 w_k1_AXP_s110
                                 + 0.0201669006670243 w_k1_AXP_s111
                                 + 0.0191112838470233 w_k1_AXP_s112
                                 + 0.0184098888720934 w_k1_AXP_s113
                                 + 0.0192464303788986 w_k1_AXP_s114
                                 + 0.0173283932128081 w_k1_AXP_s115
                                 + 0.0140177524805662 w_k1_AXP_s116
                                 + 0.00706062138473659 w_k1_AXP_s117
                                 + 0.0625243323694457 w_k1_DIS_s000
                                 + 0.0595563963592615 w_k1_DIS_s001
                                 + 0.054249986499504 w_k1_DIS_s002
                                 + 0.0231820246402928 w_k1_DIS_s003
                                 + 0.0936227859455968 w_k1_DIS_s004
                                 + 0.0395190503675859 w_k1_DIS_s005
                                 + 0.077693038206536 w_k1_DIS_s006
                                 + 0.0694328314025758 w_k1_DIS_s007
                                 + 0.040384626868232 w_k1_DIS_s010
                                 + 0.0676426407577895 w_k1_DIS_s011
                                 + 0.0367943900782878 w_k1_DIS_s012
                                 + 0.0407639287055209 w_k1_DIS_s013
                                 + 0.0527866358175899 w_k1_DIS_s014
                                 + 0.0443924788760526 w_k1_DIS_s015
                                 + 0.0410781100034882 w_k1_DIS_s016
                                 + 0.0248280960800614 w_k1_DIS_s017
                                 + 0.00793429144542243 w_k1_DIS_s100
                                 + 0.0129470203113083 w_k1_DIS_s101
                                 + 0.00810139290918138 w_k1_DIS_s102
                                 + 0.0103702306469812 w_k1_DIS_s103
                                 + 0.0132636817160159 w_k1_DIS_s104
                                 + 0.0148349879429415 w_k1_DIS_s105
                                 + 0.0141208712576835 w_k1_DIS_s106
                                 + 0.00642646983331368 w_k1_DIS_s107
                                 + 0.0153561942634974 w_k1_DIS_s110
                                 + 0.0209592787013613 w_k1_DIS_s111
                                 + 0.0208984320681624 w_k1_DIS_s112
                                 + 0.0174510826670396 w_k1_DIS_s113
                                 + 0.019476024819397 w_k1_DIS_s114
                                 + 0.0185482982103257 w_k1_DIS_s115
                                 + 0.0123661102757757 w_k1_DIS_s116
                                 + 0.00623632312404686 w_k1_DIS_s117
                                 + 0.0625095225656516 w_k1_FDX_s000
                                 + 0.0598241216950435 w_k1_FDX_s001
                                 + 0.0488934125532218 w_k1_FDX_s002
                                 + 0.0250097154730571 w_k1_FDX_s003
                                 + 0.0887582338566549 w_k1_FDX_s004
                                 + 0.0385443874399818 w_k1_FDX_s005
                                 + 0.0783806218554783 w_k1_FDX_s006
                                 + 0.0622675040311531 w_k1_FDX_s007
                                 + 0.0408435918547636 w_k1_FDX_s010
                                 + 0.068205152002685 w_k1_FDX_s011
                                 + 0.0344467776394894 w_k1_FDX_s012
                                 + 0.0421364183132186 w_k1_FDX_s013
                                 + 0.047840327091082 w_k1_FDX_s014
                                 + 0.0447766739310742 w_k1_FDX_s015
                                 + 0.0384528174592041 w_k1_FDX_s016
                                 + 0.0245965161850966 w_k1_FDX_s017
                                 + 0.007689034366792 w_k1_FDX_s100
                                 + 0.0117646280453365 w_k1_FDX_s101
                                 + 0.00763433080645986 w_k1_FDX_s102
                                 + 0.00966067116826685 w_k1_FDX_s103
                                 + 0.0136530921105692 w_k1_FDX_s104
                                 + 0.0141715176579497 w_k1_FDX_s105
                                 + 0.0129822820043075 w_k1_FDX_s106
                                 + 0.00571891605047563 w_k1_FDX_s107
                                 + 0.01438299453463 w_k1_FDX_s110
                                 + 0.0203186348753923 w_k1_FDX_s111
                                 + 0.0202573663475186 w_k1_FDX_s112
                                 + 0.0175794252455759 w_k1_FDX_s113
                                 + 0.0190043101669066 w_k1_FDX_s114
                                 + 0.0168611077974886 w_k1_FDX_s115
                                 + 0.0133466810927513 w_k1_FDX_s116
                                 + 0.00610391563924915 w_k1_FDX_s117
                                 + 0.0635035179604277 w_k1_IBM_s000
                                 + 0.0625828402626975 w_k1_IBM_s001
                                 + 0.0506189848177171 w_k1_IBM_s002
                                 + 0.0253230218641083 w_k1_IBM_s003
                                 + 0.0862416060203428 w_k1_IBM_s004
                                 + 0.0382625680875642 w_k1_IBM_s005
                                 + 0.0811612815878979 w_k1_IBM_s006
                                 + 0.0638346217981503 w_k1_IBM_s007
                                 + 0.0411844553862104 w_k1_IBM_s010
                                 + 0.0689193172302442 w_k1_IBM_s011
                                 + 0.0329311413434307 w_k1_IBM_s012
                                 + 0.0432575821520675 w_k1_IBM_s013
                                 + 0.0476711205426815 w_k1_IBM_s014
                                 + 0.045151699754339 w_k1_IBM_s015
                                 + 0.0374599961209291 w_k1_IBM_s016
                                 + 0.0247405436720215 w_k1_IBM_s017
                                 + 0.00807962361121204 w_k1_IBM_s100
                                 + 0.0121522593274728 w_k1_IBM_s101
                                 + 0.00792333164161843 w_k1_IBM_s102
                                 + 0.00997732122002952 w_k1_IBM_s103
                                 + 0.013349488780974 w_k1_IBM_s104
                                 + 0.0139903715026024 w_k1_IBM_s105
                                 + 0.013526981683559 w_k1_IBM_s106
                                 + 0.00603021001768944 w_k1_IBM_s107
                                 + 0.0158907233700202 w_k1_IBM_s110
                                 + 0.0214802456887969 w_k1_IBM_s111
                                 + 0.0206737740304734 w_k1_IBM_s112
                                 + 0.018771073745499 w_k1_IBM_s113
                                 + 0.0197628248070176 w_k1_IBM_s114
                                 + 0.0176735995604266 w_k1_IBM_s115
                                 + 0.0133521696380541 w_k1_IBM_s116
                                 + 0.00671364624561088 w_k1_IBM_s117
                                 + 0.0667882112270905 w_k1_INTC_s000
                                 + 0.0562491939483109 w_k1_INTC_s001
                                 + 0.0441247152376417 w_k1_INTC_s002
                                 + 0.0230456125372386 w_k1_INTC_s003
                                 + 0.0955598894287019 w_k1_INTC_s004
                                 + 0.041062021211754 w_k1_INTC_s005
                                 + 0.074477786304661 w_k1_INTC_s006
                                 + 0.0611964094258028 w_k1_INTC_s007
                                 + 0.0395531836917601 w_k1_INTC_s010
                                 + 0.0627915101436458 w_k1_INTC_s011
                                 + 0.0336724962646717 w_k1_INTC_s012
                                 + 0.0422532368619666 w_k1_INTC_s013
                                 + 0.048284526859711 w_k1_INTC_s014
                                 + 0.0390478759219055 w_k1_INTC_s015
                                 + 0.0357733200320049 w_k1_INTC_s016
                                 + 0.0229869964542258 w_k1_INTC_s017
                                 + 0.0083180140199108 w_k1_INTC_s100
                                 + 0.0131137721838004 w_k1_INTC_s101
                                 + 0.00721838778047875 w_k1_INTC_s102
                                 + 0.00930823257898641 w_k1_INTC_s103
                                 + 0.0141106801638448 w_k1_INTC_s104
                                 + 0.0128291648443488 w_k1_INTC_s105
                                 + 0.0133072929844275 w_k1_INTC_s106
                                 + 0.00588352927612319 w_k1_INTC_s107
                                 + 0.0144778762160976 w_k1_INTC_s110
                                 + 0.0205200758747176 w_k1_INTC_s111
                                 + 0.0184348518177701 w_k1_INTC_s112
                                 + 0.0171697373040747 w_k1_INTC_s113
                                 + 0.0211366801097372 w_k1_INTC_s114
                                 + 0.0174259594251675 w_k1_INTC_s115
                                 + 0.0116476977318525 w_k1_INTC_s116
                                 + 0.00709771321980123 w_k1_INTC_s117
                                 + 0.0621112372355906 w_k1_KO_s000
                                 + 0.0579832591711351 w_k1_KO_s001
                                 + 0.0461321486174063 w_k1_KO_s002
                                 + 0.026384754100316 w_k1_KO_s003
                                 + 0.0908603453311966 w_k1_KO_s004
                                 + 0.0393463442293248 w_k1_KO_s005
                                 + 0.0812348081786948 w_k1_KO_s006
                                 + 0.0633039443931383 w_k1_KO_s007
                                 + 0.042404810198908 w_k1_KO_s010
                                 + 0.0654380222151624 w_k1_KO_s011
                                 + 0.0346111376987699 w_k1_KO_s012
                                 + 0.0416179059051839 w_k1_KO_s013
                                 + 0.05553288263234 w_k1_KO_s014
                                 + 0.0418735392085356 w_k1_KO_s015
                                 + 0.0370380238495799 w_k1_KO_s016
                                 + 0.0227172091487528 w_k1_KO_s017
                                 + 0.00770557956958335 w_k1_KO_s100
                                 + 0.0126717018649819 w_k1_KO_s101
                                 + 0.00773461990192778 w_k1_KO_s102
                                 + 0.00992892010038959 w_k1_KO_s103
                                 + 0.0142233218562644 w_k1_KO_s104
                                 + 0.0124585849747166 w_k1_KO_s105
                                 + 0.0123508083632178 w_k1_KO_s106
                                 + 0.00581558292175923 w_k1_KO_s107
                                 + 0.0145265822785652 w_k1_KO_s110
                                 + 0.021671609923304 w_k1_KO_s111
                                 + 0.0198592505462544 w_k1_KO_s112
                                 + 0.0171672783492953 w_k1_KO_s113
                                 + 0.0201972188730233 w_k1_KO_s114
                                 + 0.0184368562708173 w_k1_KO_s115
                                 + 0.0115584930744644 w_k1_KO_s116
                                 + 0.00622581550147382 w_k1_KO_s117
                                 + 0.0557298480151827 w_k1_MAT_s000
                                 + 0.0656274867981166 w_k1_MAT_s001
                                 + 0.0433797552022217 w_k1_MAT_s002
                                 + 0.027761353247147 w_k1_MAT_s003
                                 + 0.0799622255747707 w_k1_MAT_s004
                                 + 0.0360112920346266 w_k1_MAT_s005
                                 + 0.0679732978941144 w_k1_MAT_s006
                                 + 0.0563323970872425 w_k1_MAT_s007
                                 + 0.0524974880990836 w_k1_MAT_s010
                                 + 0.0653072701969744 w_k1_MAT_s011
                                 + 0.0373393384332823 w_k1_MAT_s012
                                 + 0.0521481399849103 w_k1_MAT_s013
                                 + 0.0472134001131769 w_k1_MAT_s014
                                 + 0.0446881107914345 w_k1_MAT_s015
                                 + 0.0404556368173101 w_k1_MAT_s016
                                 + 0.0233874525922344 w_k1_MAT_s017
                                 + 0.00907933090281848 w_k1_MAT_s100
                                 + 0.00909700154823557 w_k1_MAT_s101
                                 + 0.00673899508499233 w_k1_MAT_s102
                                 + 0.0116724479166748 w_k1_MAT_s103
                                 + 0.0150062009791874 w_k1_MAT_s104
                                 + 0.0157258769799129 w_k1_MAT_s105
                                 + 0.0137469436220213 w_k1_MAT_s106
                                 + 0.00675957020573778 w_k1_MAT_s107
                                 + 0.0122475702743335 w_k1_MAT_s110
                                 + 0.0272442485172488 w_k1_MAT_s111
                                 + 0.0185902597677118 w_k1_MAT_s112
                                 + 0.0234319757158193 w_k1_MAT_s113
                                 + 0.0193636690023501 w_k1_MAT_s114
                                 + 0.0171602207713216 w_k1_MAT_s115
                                 + 0.0122412097927982 w_k1_MAT_s116
                                 + 0.00643137955608189 w_k1_MAT_s117
                                 + 0.0561945285704537 w_k1_MSFT_s000
                                 + 0.0602888443163896 w_k1_MSFT_s001
                                 + 0.0485407527783246 w_k1_MSFT_s002
                                 + 0.0243371335426274 w_k1_MSFT_s003
                                 + 0.0864675316839592 w_k1_MSFT_s004
                                 + 0.0369728650649923 w_k1_MSFT_s005
                                 + 0.0723766434510663 w_k1_MSFT_s006
                                 + 0.0589382777762765 w_k1_MSFT_s007
                                 + 0.0438265328471693 w_k1_MSFT_s010
                                 + 0.0641922405115096 w_k1_MSFT_s011
                                 + 0.0339030033622425 w_k1_MSFT_s012
                                 + 0.0364042088425582 w_k1_MSFT_s013
                                 + 0.0499400769662334 w_k1_MSFT_s014
                                 + 0.0416126449545642 w_k1_MSFT_s015
                                 + 0.0387344597549169 w_k1_MSFT_s016
                                 + 0.0238843062716891 w_k1_MSFT_s017
                                 + 0.008062587298757 w_k1_MSFT_s100
                                 + 0.0116236077648647 w_k1_MSFT_s101
                                 + 0.00734727957517269 w_k1_MSFT_s102
                                 + 0.00959276728391274 w_k1_MSFT_s103
                                 + 0.0144822027776825 w_k1_MSFT_s104
                                 + 0.0137333922318886 w_k1_MSFT_s105
                                 + 0.0129565224126493 w_k1_MSFT_s106
                                 + 0.00585964185901814 w_k1_MSFT_s107
                                 + 0.014815315736591 w_k1_MSFT_s110
                                 + 0.0222384589304085 w_k1_MSFT_s111
                                 + 0.0192507816162144 w_k1_MSFT_s112
                                 + 0.0148994213514768 w_k1_MSFT_s113
                                 + 0.0205045560348712 w_k1_MSFT_s114
                                 + 0.0160865188946635 w_k1_MSFT_s115
                                 + 0.0113665706480207 w_k1_MSFT_s116
                                 + 0.0061146004681066 w_k1_MSFT_s117
                                 + 0.0600626801715799 w_k1_PG_s000
                                 + 0.0598506876348716 w_k1_PG_s001
                                 + 0.0507408536614097 w_k1_PG_s002
                                 + 0.0244353573166935 w_k1_PG_s003
                                 + 0.0920645031232794 w_k1_PG_s004
                                 + 0.0359369992964654 w_k1_PG_s005
                                 + 0.0746586903885155 w_k1_PG_s006
                                 + 0.0658206648785172 w_k1_PG_s007
                                 + 0.0421912318499443 w_k1_PG_s010
                                 + 0.0685253639580292 w_k1_PG_s011
                                 + 0.0330679184733625 w_k1_PG_s012
                                 + 0.0421381727419916 w_k1_PG_s013
                                 + 0.0501481466548798 w_k1_PG_s014
                                 + 0.04389464942733 w_k1_PG_s015
                                 + 0.0387798374589309 w_k1_PG_s016
                                 + 0.0220644363171551 w_k1_PG_s017
                                 + 0.00860309074912304 w_k1_PG_s100
                                 + 0.0120165781674397 w_k1_PG_s101
                                 + 0.0070040587482647 w_k1_PG_s102
                                 + 0.0101845980488507 w_k1_PG_s103
                                 + 0.0141069620716556 w_k1_PG_s104
                                 + 0.0155175992839731 w_k1_PG_s105
                                 + 0.012542442982081 w_k1_PG_s106
                                 + 0.00571984268672907 w_k1_PG_s107
                                 + 0.0144034731586314 w_k1_PG_s110
                                 + 0.020119534100926 w_k1_PG_s111
                                 + 0.0200208179414368 w_k1_PG_s112
                                 + 0.017596857280314 w_k1_PG_s113
                                 + 0.0204649009366731 w_k1_PG_s114
                                 + 0.0165609898180427 w_k1_PG_s115
                                 + 0.0130741741412744 w_k1_PG_s116
                                 + 0.00596089187607071 w_k1_PG_s117 >= 1.19
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
