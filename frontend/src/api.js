export const getProtectedData = async () => {
    const token = localStorage.getItem('access');
  
    try {
      const response = await fetch('http://localhost:8000/api/tu-endpoint/', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
  
      if (!response.ok) {
        throw new Error('No autorizado');
      }
  
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching protected data:', error);
    }
  };
    