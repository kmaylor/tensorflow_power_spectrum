class PowerSpectrum(object):
    """ Class for calculating a the power spectrum (1D or 2D) of an image in tensorflow.
        Expects square image as input.
    """
    def __init__(self,image_size=None):
        """image_size: only needed
        """
        self.image_size = image_size
        if image_size is not None:
            self.az_mask=build_azimuthal_mask()

    def power2D(self,x):
        x = tf.spectral.fft2d(tf.cast(x,dtype=tf.complex64))
        x = tf.cast(x,dtype=tf.complex64)
        xl,xu = tf.split(x,2,axis=1)
        xll,xlr = tf.split(xl,2,axis=2)
        xul,xur = tf.split(xu,2,axis=2)
        xu = tf.concat([xlr,xll],axis=2)
        xl = tf.concat([xur,xul],axis=2)
        x=tf.concat([xl,xu],axis=1)
        x = tf.abs(x)
        return tf.square(x)
     
    def build_azimuthal_mask(self):
        
        x,y = np.meshgrid(np.arange(self.image_size),np.arange(self.image_size))
        R = np.sqrt((x-self.image_size/2)**2+(y-self.image_size/2)**2)
        masks = np.array(list(map(lambda r : (R >= r-.5) & (R < r+.5),np.arange(1,int(self.image_size/2+1),1))))
        norm = np.sum(masks,axis=(1,2),keepdims=True)
        masks=masks/norm
        n=len(masks)
        return tf.reshape(tf.cast(masks,dtype=tf.float32),(1,n,self.image_size,self.image_size))
        
    def az_average(self,x):
        x=tf.reshape(x,(-1,1,self.image_size,self.image_size))
        return tf.reduce_sum(tf.reduce_sum(tf.multiply(self.az_mask,x),axis=3),axis=2)
    
    def power1D(self,x):
        x = self.power2D(x)
        az_avg = self.az_average(x)
        ell=np.arange(int(az_avg.shape[1]))*9
        return tf.multiply(az_avg,tf.reshape(tf.cast(ell*(ell+1)/2/np.pi,dtype=tf.float32),(1,-1)))

    def build_model(self):
        input_img = Input(shape=(self.image_size,self.image_size))
        if image_size is not None:
            p=Lambda(lambda x: power1D(x))(input_img) 
        else:
            p=Lambda(lambda x: power2D(x))(input_img)
        self.model=Model(inputs=input_img,outputs=p)

    def __call__(self,image):
        return self.model.predict(image)




